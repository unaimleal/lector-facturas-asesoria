import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from pypdf import PdfReader, PdfWriter
import io
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Extractor Facturas AI", layout="wide")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- FUNCIONES DE LÓGICA (BACKEND) ---

def dividir_pdf_en_memoria(uploaded_file):
    """
    Recibe un archivo subido por Streamlit y lo divide en bytes en memoria.
    No guarda nada en disco duro para mayor velocidad y privacidad.
    """
    reader = PdfReader(uploaded_file)
    archivos_individuales = []

    # Si es 1 página, devolvemos el objeto tal cual
    if len(reader.pages) == 1:
        # Resetear puntero del archivo para leerlo desde el inicio
        uploaded_file.seek(0)
        return [("factura_unica", uploaded_file)]

    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        
        # Guardar en un buffer de memoria (RAM) en lugar de disco
        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        output_buffer.seek(0) # Volver al inicio del buffer
        
        nombre_temp = f"pag_{i+1}_{uploaded_file.name}"
        archivos_individuales.append((nombre_temp, output_buffer))
    
    return archivos_individuales

def procesar_con_gemini(archivo_bytes, modelo_nombre):
    # Usamos gemini-1.5-flash (el 2.5 aún no es público estable para API general)
    model = genai.GenerativeModel(
        modelo_nombre,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = """
    Analiza esta factura.
    IMPORTANTE:
    1. 'nif_emisor': NIF de quien EMITE la factura (quien cobra, muy importante).
    2. 'nombre_proveedor': Nombre fiscal de la empresa emisora.
    3. Fechas: DD/MM/AAAA.
    4. tipo_iva: en Porcentaje pero sin el símbolo de %
    5. Devuelve JSON válido.
    
    Campos:
    {
        "nif_emisor": "string",
        "fecha": "string",
        "numero_factura": "string",
        "nombre_proveedor": "string",
        "base_imponible": float,
        "cuota_iva": float,
        "tipo_iva": float,
        "total_factura": float
    }
    """

    try:
        # Streamlit entrega BytesIO, Gemini a veces prefiere guardar en temp, 
        # pero la forma más limpia en web apps es pasar los bytes directamente si la librería lo soporta,
        # o subirlo usando upload_file desde un path. 
        # Para simplificar en Streamlit Cloud sin gestionar archivos temporales complejos:
        
        # 1. Guardamos temporalmente el buffer en un archivo físico efímero
        with open("temp_upload.pdf", "wb") as f:
            f.write(archivo_bytes.getbuffer())
        
        # 2. Subimos a Gemini
        sample_file = genai.upload_file(path="temp_upload.pdf", display_name="Factura")
        
        # Esperar a que se procese
        while sample_file.state.name == "PROCESSING":
            time.sleep(1)
            sample_file = genai.get_file(sample_file.name)

        response = model.generate_content([sample_file, prompt])
        datos = json.loads(response.text)
        
        # Limpieza nube
        sample_file.delete()
        return datos

    except Exception as e:
        st.error(f"Error procesando un archivo: {e}")
        return None

# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📄 Asesoría AI - Procesador de Facturas")
st.markdown("Arrastra tus PDFs y genera el Excel automáticamente.")

# 1. BARRA LATERAL: Configuración
with st.sidebar:
    st.header("Configuración")
    modelo = st.selectbox("Modelo AI", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.info("Nota: Flash es más rápido y barato. Pro es más preciso.")

# 2. ZONA PRINCIPAL: Entradas
col1, col2 = st.columns(2)
with col1:
    # Input numérico para el contador interno
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
        
        # Aplanar lista de PDFs (por si hay multipágina)
        lista_trabajo = []
        estado.write("Analizando páginas de los PDFs...")
        
        for archivo in uploaded_files:
            sub_pdfs = dividir_pdf_en_memoria(archivo)
            lista_trabajo.extend(sub_pdfs)
            
        total_docs = len(lista_trabajo)
        
        # Procesar cada factura
        for i, (nombre, buffer) in enumerate(lista_trabajo):
            contador_interno += 1 # Incrementamos el número interno
            estado.write(f"Procesando {i+1}/{total_docs}: {nombre}...")
            
            datos = procesar_con_gemini(buffer, modelo)
            
            if datos:
                # Añadir el número interno calculado
                datos['numero_interno'] = contador_interno
                datos['archivo_origen'] = nombre
                datos_totales.append(datos)
            
            # Actualizar barra
            progreso.progress((i + 1) / total_docs)
            
        estado.write("¡Proceso completado! Generando Excel...")
        
        # CREAR DATAFRAME Y EXCEL
        
        if datos_totales:
            df = pd.DataFrame(datos_totales)
            
            if 'nif_emisor' in df.columns:
                df['NIF'] = df['nif_emisor'].astype(str).str.replace('-', '', regex=False).str.replace('ES', '', regex=False)
            else:
                df['NIF'] = ""

            # 1. Definimos todas las columnas que queremos (sin repetir)
            # Quitamos 'numero_interno' de la lista de 'cols_deseadas' para que no se duplique abajo
            columnas_base = ["nombre_proveedor", "fecha", "numero_factura", "base_imponible", "tipo_iva", "cuota_iva", "total_factura"]
            for col in columnas_base:
                if col not in df.columns:
                    df[col] = ""
            
            if archivo_maestro is not None:
                try:
                    df_maestro = pd.read_excel(archivo_maestro)
                    # Limpiamos también el NIF del maestro para que coincidan
                    df_maestro['NIF'] = df_maestro['NIF'].astype(str).str.replace('-', '', regex=False).str.replace('ES', '', regex=False).str.strip()
                    
                    # Seleccionamos solo las columnas necesarias del maestro para evitar basura
                    # Asumimos que el Excel tiene columnas llamadas 'Cuenta' y 'Contrapartida'
                    df_maestro = df_maestro[['NIF', 'Cuenta', 'Contrapartida']]
                    
                    # Unimos las tablas por la columna NIF
                    df = pd.merge(df, df_maestro, on='NIF', how='left')
                    
                    # Renombramos las columnas según tu petición
                    df = df.rename(columns={
                        'Cuenta': 'COD.PROVEED',
                        'Contrapartida': 'COD. GASTOS'
                    })
                except Exception as e:
                    st.error(f"Error al cruzar con el Excel maestro: {e}. Asegúrate de que el Excel tenga las columnas 'NIF', 'Cuenta' y 'Contrapartida'.")


            columnas_nuevas = ['COD.PROVEED', 'COD. GASTOS']
            for col in columnas_nuevas:
                if col not in df.columns:
                    df[col] = ""

            # 4. Crear columna SECCION (siempre nula)
            df['SECCION'] = ""

            # 5. Formatear COMENTARIO (Barra vertical y nuevo orden)
            df['COMENTARIO'] = df.apply(
                lambda x: f"{str(x.get('numero_interno', ''))} | {str(x.get('numero_factura', ''))} | {str(x.get('nombre_proveedor', ''))}", 
                axis=1
            )

            # 6. RENOMBRAR columnas restantes para el formato final
            df = df.rename(columns={
                'fecha': 'FECHA',
                'numero_factura': 'Nº FRA.',
                'total_factura': 'TOTAL FRA.',
                'tipo_iva': 'TIPO IVA',
                'base_imponible': 'BASE IMP.',
                'cuota_iva': 'CUOTA IVA'
            })

            # 7. ORDEN FINAL Y SELECCIÓN ESTRICTA
            # Solo se incluyen estas columnas y en este orden
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

            # Reindexamos para quedarnos solo con lo que pediste (si alguna falta, se crea vacía)
            df_final = df.reindex(columns=orden_final)

            st.success("Análisis y cruce completado.")
            st.dataframe(df_final)
            
            # Botón de descarga
            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer_excel.getvalue(),
                file_name=f"facturas_procesadas_desde_{ultimo_numero + 1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("No se pudieron extraer datos de ningún archivo.")