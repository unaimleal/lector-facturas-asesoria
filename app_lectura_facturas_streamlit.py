import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
# Mantenemos pypdf por si en el futuro quieres volver a contar páginas, 
# pero ya no se usa para dividir.
from pypdf import PdfReader 
import io
import time
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Extractor Facturas AI", layout="wide")

# Intentamos cargar la API Key de los secretos
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("No se encontró la API KEY en los secretos. Configúrala en Streamlit Cloud.")

# --- FUNCIONES DE LÓGICA (BACKEND) ---

def procesar_con_gemini(uploaded_file, modelo_nombre):
    """
    Sube el archivo PDF completo a Gemini y extrae los datos.
    """
    model = genai.GenerativeModel(
        modelo_nombre,
        generation_config={"response_mime_type": "application/json"}
    )

    # --- PROMPT ACTUALIZADO PARA GESTIONAR IVAS MÚLTIPLES ---
    prompt = """
    Analiza este documento PDF completo (puede tener varias páginas) como UNA SOLA factura.
    
    INSTRUCCIONES CRÍTICAS SOBRE IMPUESTOS:
    1. Si la factura tiene UN SOLO tipo de IVA: Extrae base, cuota y tipo normalmente.
    2. Si la factura tiene VARIOS tipos de IVA (ej: 21% y 10%, o 21% y Exento/0%):
       DEBES devolver 'base_imponible', 'cuota_iva' y 'tipo_iva' como null (nulo).
       Sin embargo, SÍ debes extraer la 'fecha', el 'numero_factura' y el 'total_factura'.

    OTROS CAMPOS:
    - 'nif_emisor': NIF de quien EMITE la factura.
    - 'nombre_proveedor': Nombre fiscal de la empresa emisora.
    - Fechas: DD/MM/AAAA.
    
    Devuelve JSON estrictamente válido:
    {
        "nif_emisor": "string",
        "fecha": "string",
        "numero_factura": "string",
        "nombre_proveedor": "string",
        "base_imponible": float or null,
        "cuota_iva": float or null,
        "tipo_iva": float or null,
        "total_factura": float
    }
    """

    try:
        # 1. Guardar temporalmente el archivo en disco (Gemini necesita path o subida directa)
        # Usamos un nombre temporal seguro
        temp_filename = f"temp_{int(time.time())}.pdf"
        
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Subir el archivo a la API de Gemini
        sample_file = genai.upload_file(path=temp_filename, display_name="Factura")
        
        # Esperar a que se procese (estado ACTIVE)
        while sample_file.state.name == "PROCESSING":
            time.sleep(1)
            sample_file = genai.get_file(sample_file.name)

        # 3. Generar la respuesta
        response = model.generate_content([sample_file, prompt])
        datos = json.loads(response.text)
        
        # 4. Limpieza (Borrar archivo de la nube y del disco local)
        sample_file.delete()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return datos

    except Exception as e:
        st.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
        return None

# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📄 Asesoría AI - Procesador de Facturas")
st.markdown("Arrastra tus PDFs. **Nota:** Cada archivo PDF se cuenta como una única factura.")

# 1. BARRA LATERAL: Configuración
with st.sidebar:
    st.header("Configuración")
    # Nota: Gemini 1.5 es el estable actual para archivos. 
    modelo = st.selectbox("Modelo AI", ["gemini-1.5-flash", "gemini-1.5-pro"])
    st.info("Flash es rápido. Pro es más preciso con facturas complejas.")

# 2. ZONA PRINCIPAL: Entradas
col1, col2 = st.columns(2)
with col1:
    ultimo_numero = st.number_input(
        "Último número interno registrado", 
        min_value=0, 
        value=1000, 
        step=1,
        help="La primera factura de este lote será este número + 1"
    )

    archivo_maestro = st.file_uploader("Subir Maestro Proveedores (Excel)", type=["xlsx", "xls"])

with col2:
    uploaded_files = st.file_uploader(
        "Sube las facturas (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

# 3. BOTÓN DE PROCESO
if st.button("🚀 Procesar Facturas", type="primary"):
    if not uploaded_files:
        st.warning("⚠️ No has subido ningún archivo.")
    else:
        # Barra de progreso
        progreso = st.progress(0)
        estado = st.empty()
        
        datos_totales = []
        contador_interno = ultimo_numero
        
        total_docs = len(uploaded_files)
        
        # --- BUCLE DE PROCESAMIENTO (1 PDF = 1 FACTURA) ---
        for i, archivo in enumerate(uploaded_files):
            contador_interno += 1 
            estado.write(f"Procesando {i+1}/{total_docs}: {archivo.name}...")
            
            # Llamamos directamente con el archivo (sin dividir páginas)
            datos = procesar_con_gemini(archivo, modelo)
            
            if datos:
                datos['numero_interno'] = contador_interno
                datos['archivo_origen'] = archivo.name
                datos_totales.append(datos)
            
            # Actualizar barra
            progreso.progress((i + 1) / total_docs)
            
        estado.write("¡Proceso completado! Generando Excel...")
        
        # --- CREACIÓN DEL DATAFRAME FINAL ---
        
        if datos_totales:
            df = pd.DataFrame(datos_totales)
            
            if 'nif_emisor' in df.columns:
                df['NIF'] = df['nif_emisor'].astype(str).str.replace('-', '', regex=False).str.replace('ES', '', regex=False).str.strip().str.upper()
                df['NIF'] = df['NIF'].replace({'NONE': '', 'NAN': ''})
            else:
                df['NIF'] = ""

            columnas_base = ["nombre_proveedor", "fecha", "numero_factura", "base_imponible", "tipo_iva", "cuota_iva", "total_factura"]
            for col in columnas_base:
                if col not in df.columns:
                    df[col] = ""
            
            if archivo_maestro is not None:
                try:
                    df_maestro = pd.read_excel(archivo_maestro)
                    df_maestro['NIF'] = df_maestro['NIF'].astype(str).str.replace(r'\.0$', '', regex=True) # Quitar decimales excel
                    df_maestro['NIF'] = df_maestro['NIF'].str.replace('-', '', regex=False).str.replace('ES', '', regex=False).str.strip().str.upper()
                    
                    if 'Cuenta' in df_maestro.columns and 'Contrapartida' in df_maestro.columns:
                        df_maestro = df_maestro[['NIF', 'Cuenta', 'Contrapartida']]
                        df = pd.merge(df, df_maestro, on='NIF', how='left')
                        df = df.rename(columns={'Cuenta': 'COD.PROVEED', 'Contrapartida': 'COD. GASTOS'})
                    else:
                        st.error("El Excel maestro debe tener columnas: NIF, Cuenta, Contrapartida")
                except Exception as e:
                    st.error(f"Error al cruzar con el Excel maestro: {e}")

            columnas_nuevas = ['COD.PROVEED', 'COD. GASTOS']
            for col in columnas_nuevas:
                if col not in df.columns:
                    df[col] = ""

            df['SECCION'] = ""

            df['COMENTARIO'] = df.apply(
                lambda x: f"{str(x.get('numero_interno', ''))} | {str(x.get('numero_factura', ''))} | {str(x.get('nombre_proveedor', ''))}", 
                axis=1
            )

            df = df.rename(columns={
                'fecha': 'FECHA',
                'numero_factura': 'Nº FRA.',
                'total_factura': 'TOTAL FRA.',
                'tipo_iva': 'TIPO IVA',
                'base_imponible': 'BASE IMP.',
                'cuota_iva': 'CUOTA IVA'
            })

            orden_final = [
                "FECHA",
                "COD.PROVEED",
                "Nº FRA.",
                "COMENTARIO",
                "TOTAL FRA.",
                "COD. GASTOS",
                "TIPO IVA",
                "SECCION",
                "BASE IMP.",
                "CUOTA IVA"
            ]

            # Reindexamos: Esto descarta todo lo que no esté en la lista y ordena
            df_final = df.reindex(columns=orden_final)
            
            # Rellenar los valores nulos (NaN) con cadena vacía para que el Excel quede limpio
            df_final = df_final.fillna("")

            st.success("Análisis y cruce completado.")
            st.dataframe(df_final)
            
            # --- DESCARGA DEL ARCHIVO (AHORA COINCIDE CON LO QUE VES) ---
            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
                df_final.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer_excel.getvalue(),
                file_name=f"facturas_procesadas_desde_{ultimo_numero + 1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("No se pudieron extraer datos de ningún archivo.")