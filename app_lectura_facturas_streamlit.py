import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Extractor Facturas AI", layout="wide")

# Modelos disponibles (nombre en pantalla -> id de la API)
MODELOS = {
    "Económico y rápido · Gemini 3.5 Flash-Lite (recomendado)": "gemini-3.5-flash-lite",
    "Más preciso · Gemini 3.8 Flash": "gemini-3.8-flash",
}

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    client = None
    st.error("No se encontró la API KEY en los secretos. Configúrala en Streamlit Cloud.")

PROMPT = """
Analiza este documento PDF completo (puede tener varias páginas) como UNA SOLA factura.

INSTRUCCIONES CRÍTICAS SOBRE IMPUESTOS:
1. Si la factura tiene UN SOLO tipo de IVA: Extrae base, cuota y tipo normalmente.
2. Si la factura tiene VARIOS tipos de IVA (ej: 21% y 10%, o 21% y Exento/0%):
   DEBES devolver 'base_imponible', 'cuota_iva', 'fecha' y 'tipo_iva' como null (nulo).
   Sin embargo, SÍ debes extraer el 'nombre_proveedor', el 'numero_factura' y el 'total_factura'.

OTROS CAMPOS:
- 'nif_emisor': NIF de quien EMITE la factura, sin guiones, barras o puntos, solo las letras y numeros
- 'nombre_proveedor': Nombre fiscal de la empresa emisora.
- Fechas: DD/MM/AAAA.

Devuelve JSON estrictamente válido:
{
    "nif_emisor": "string",
    "fecha": "string" or null,
    "numero_factura": "string",
    "nombre_proveedor": "string",
    "base_imponible": float or null,
    "cuota_iva": float or null,
    "tipo_iva": float or null,
    "total_factura": float
}
"""


# --- FUNCIONES DE LÓGICA (BACKEND) ---

def procesar_con_gemini(uploaded_file, modelo_id):
    """Envía el PDF completo a Gemini y devuelve los datos de la factura."""
    try:
        pdf_part = types.Part.from_bytes(
            data=uploaded_file.getvalue(), mime_type="application/pdf"
        )
        response = client.models.generate_content(
            model=modelo_id,
            contents=[pdf_part, PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        datos = json.loads(response.text)
        if isinstance(datos, list):  # por si el modelo devuelve una lista
            datos = datos[0]
        return datos
    except Exception as e:
        st.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
        return None


def limpiar_nif(serie):
    return (
        serie.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("ES", "", regex=False)
        .str.strip()
        .str.upper()
    )


def plantilla_maestro():
    df = pd.DataFrame(
        {
            "NIF": ["B12345678", "A87654321"],
            "Cuenta": [4000001, 4000002],
            "Contrapartida": [6290001, 6220001],
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📄 Asesoría AI - Procesador de Facturas")

with st.expander("ℹ️ ¿Qué tengo que subir? (guía rápida)", expanded=True):
    st.markdown(
        """
**Necesitas 3 cosas:**

**1. Último número interno registrado** (campo numérico)
Es el último número de registro que ya usasteis en la contabilidad. La primera factura de este lote
será ese número **+ 1**, y las siguientes irán correlativas.

**2. Maestro de proveedores (Excel)** — *opcional pero recomendable*
Un Excel (`.xlsx`, `.xls`) o `.csv` con **una fila por proveedor** y **exactamente estas 3 columnas**
(en la primera fila, con estos nombres):

| Columna | Qué contiene | Ejemplo |
|---|---|---|
| `NIF` | NIF/CIF del proveedor (da igual si lleva guiones, espacios o "ES") | B12345678 |
| `Cuenta` | Código de la cuenta del proveedor → saldrá como **COD.PROVEED** | 4000001 |
| `Contrapartida` | Cuenta de gasto habitual → saldrá como **COD. GASTOS** | 6290001 |

La app cruza el NIF que lee de cada factura con este maestro. Si un proveedor no está en el maestro,
sus códigos saldrán **vacíos** (te avisamos al terminar).

**3. Facturas en PDF**
Sube todos los PDFs del lote. **Cada PDF = una factura** (aunque tenga varias páginas).
No metas dos facturas distintas en el mismo PDF.

**Qué obtienes:** un Excel con las columnas FECHA, COD.PROVEED, Nº FRA., COMENTARIO, TOTAL FRA.,
COD. GASTOS, TIPO IVA, SECCION, BASE IMP. y CUOTA IVA.

⚠️ **Facturas con varios tipos de IVA** (ej. 21 % y 10 %): FECHA, TIPO IVA, BASE IMP. y CUOTA IVA
salen **vacíos a propósito**; hay que rellenarlos a mano. Sí salen el nº de factura, el proveedor y el total.
        """
    )
    st.download_button(
        "📥 Descargar plantilla de ejemplo del maestro",
        data=plantilla_maestro(),
        file_name="plantilla_maestro_proveedores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# 1. BARRA LATERAL: Configuración
with st.sidebar:
    st.header("Configuración")
    modelo_label = st.selectbox("Modelo AI", list(MODELOS.keys()))
    modelo = MODELOS[modelo_label]
    st.info(
        "Flash-Lite es el más barato y suele bastar para facturas normales. "
        "Si falla con facturas complejas o escaneadas, prueba Gemini 3.8 Flash."
    )

# 2. ZONA PRINCIPAL: Entradas
col1, col2 = st.columns(2)
with col1:
    ultimo_numero = st.number_input(
        "1️⃣ Último número interno registrado",
        min_value=0,
        value=1000,
        step=1,
        help="La primera factura de este lote será este número + 1",
    )

    archivo_maestro = st.file_uploader(
        "2️⃣ Maestro de proveedores (Excel con columnas: NIF, Cuenta, Contrapartida)",
        type=["xlsx", "xls", "csv"],
    )

with col2:
    uploaded_files = st.file_uploader(
        "3️⃣ Facturas (PDF, una factura por archivo)",
        type=["pdf"],
        accept_multiple_files=True,
    )

# 3. BOTÓN DE PROCESO
if st.button("🚀 Procesar Facturas", type="primary"):
    if not uploaded_files:
        st.warning("⚠️ No has subido ningún archivo.")
    elif client is None:
        st.error("Falta la API KEY, no se puede procesar.")
    else:
        progreso = st.progress(0)
        estado = st.empty()

        datos_totales = []
        contador_interno = ultimo_numero
        total_docs = len(uploaded_files)

        for i, archivo in enumerate(uploaded_files):
            contador_interno += 1
            estado.write(f"Procesando {i+1}/{total_docs}: {archivo.name}...")

            datos = procesar_con_gemini(archivo, modelo)

            if datos:
                datos["numero_interno"] = contador_interno
                datos["archivo_origen"] = archivo.name
                datos_totales.append(datos)

            progreso.progress((i + 1) / total_docs)

        estado.write("¡Proceso completado! Generando Excel...")

        if datos_totales:
            df = pd.DataFrame(datos_totales)

            if "nif_emisor" in df.columns:
                df["NIF"] = limpiar_nif(df["nif_emisor"])
                df["NIF"] = df["NIF"].replace({"NONE": "", "NAN": ""})
            else:
                df["NIF"] = ""

            columnas_base = [
                "nombre_proveedor", "fecha", "numero_factura",
                "base_imponible", "tipo_iva", "cuota_iva", "total_factura",
            ]
            for col in columnas_base:
                if col not in df.columns:
                    df[col] = ""

            if archivo_maestro is not None:
                try:
                    if archivo_maestro.name.lower().endswith(".csv"):
                        df_maestro = pd.read_csv(archivo_maestro, sep=None, engine="python")
                    else:
                        df_maestro = pd.read_excel(archivo_maestro)

                    if {"NIF", "Cuenta", "Contrapartida"}.issubset(df_maestro.columns):
                        df_maestro["NIF"] = limpiar_nif(df_maestro["NIF"])
                        df_maestro = df_maestro[["NIF", "Cuenta", "Contrapartida"]].drop_duplicates("NIF")
                        df = pd.merge(df, df_maestro, on="NIF", how="left")
                        df = df.rename(columns={"Cuenta": "COD.PROVEED", "Contrapartida": "COD. GASTOS"})
                    else:
                        st.error("El Excel maestro debe tener las columnas: NIF, Cuenta, Contrapartida")
                except Exception as e:
                    st.error(f"Error al cruzar con el Excel maestro: {e}")

            for col in ["COD.PROVEED", "COD. GASTOS"]:
                if col not in df.columns:
                    df[col] = ""
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            # Aviso de proveedores que no estaban en el maestro
            if archivo_maestro is not None and "COD.PROVEED" in df.columns:
                sin_cod = df[df["COD.PROVEED"].isna()]
                if not sin_cod.empty:
                    st.warning(
                        f"{len(sin_cod)} factura(s) sin código de proveedor (NIF no encontrado en el maestro): "
                        + ", ".join(sin_cod["archivo_origen"].astype(str))
                    )

            df["SECCION"] = ""

            df["COMENTARIO"] = df.apply(
                lambda x: f"{x.get('numero_interno', '')} | {x.get('numero_factura', '')} | {x.get('nombre_proveedor', '')}",
                axis=1,
            )

            df = df.rename(columns={
                "fecha": "FECHA",
                "numero_factura": "Nº FRA.",
                "total_factura": "TOTAL FRA.",
                "tipo_iva": "TIPO IVA",
                "base_imponible": "BASE IMP.",
                "cuota_iva": "CUOTA IVA",
            })

            orden_final = [
                "FECHA", "COD.PROVEED", "Nº FRA.", "COMENTARIO", "TOTAL FRA.",
                "COD. GASTOS", "TIPO IVA", "SECCION", "BASE IMP.", "CUOTA IVA",
            ]
            df_final = df.reindex(columns=orden_final)

            st.success("Análisis y cruce completado.")
            st.dataframe(df_final)

            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                df_final.to_excel(writer, index=False)

            st.download_button(
                label="📥 Descargar Excel",
                data=buffer_excel.getvalue(),
                file_name=f"facturas_procesadas_desde_{ultimo_numero + 1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("No se pudieron extraer datos de ningún archivo.")
