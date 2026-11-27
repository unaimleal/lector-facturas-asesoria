import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import io
import os
import re
import csv
from copy import copy
import openpyxl

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Extractor Facturas AI", layout="wide")

# Modelos disponibles (nombre en pantalla -> id de la API)
MODELOS = {
    "Económico y rápido · Gemini 3.5 Flash-Lite (recomendado)": "gemini-3.5-flash-lite",
    "Más preciso · Gemini 3.8 Flash": "gemini-3.8-flash",
}

# Plantilla del Excel de salida (debe estar en la misma carpeta que app.py)
RUTA_PLANTILLA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantilla_entrada_apuntes.xlsx")
HOJA_COMPRAS = "Compras"
FILA_INICIO_COMPRAS = 5   # la fila 4 son las cabeceras
FILA_MODELO_ESTILO = 98   # última fila formateada de la plantilla (se copia su estilo si hay más facturas)

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
        if isinstance(datos, list):
            datos = datos[0]
        return datos
    except Exception as e:
        st.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
        return None


# --- HELPERS (sin dependencia de Streamlit) ---

def limpiar_nif(valor):
    """Deja solo letras y números en mayúsculas y quita el prefijo 'ES' si sobra."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    s = re.sub(r"[^A-Z0-9]", "", str(valor).upper())
    if s in ("NONE", "NAN", "NULL"):
        return ""
    if s.startswith("ES") and len(s) == 11:
        s = s[2:]
    return s


def _filas_crudas(archivo):
    """Lee el CSV/Excel del maestro como lista de filas (listas de texto), sin interpretar nada."""
    nombre = archivo.name.lower()
    contenido = archivo.getvalue()
    if nombre.endswith(".csv"):
        try:
            texto = contenido.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = contenido.decode("cp1252")
        delim = ";" if texto.count(";") >= texto.count(",") else ","
        return [[c.strip() for c in fila] for fila in csv.reader(io.StringIO(texto), delimiter=delim)]
    df = pd.read_excel(io.BytesIO(contenido), header=None, dtype=str).fillna("")
    return [[str(c).strip() for c in fila] for fila in df.values.tolist()]


def leer_maestro(archivo):
    """
    Lee el 'INFORME PLAN CONTABLE' tal cual sale del programa:
    busca solo la fila de cabeceras (Cuenta / NIF / Contrapartida), ignora las filas de
    arriba y la línea de guiones, y se queda con las cuentas que tienen NIF.
    Devuelve (df con NIF, Cuenta, Contrapartida, lista de NIF repetidos).
    """
    filas = _filas_crudas(archivo)
    idx_cab = None
    for i, fila in enumerate(filas):
        celdas = [c.lower() for c in fila]
        if "cuenta" in celdas and "nif" in celdas:
            idx_cab = i
            break
    if idx_cab is None:
        raise ValueError("No encuentro la fila de cabeceras (Cuenta, NIF, Contrapartida) en el fichero.")

    cab = [c.lower() for c in filas[idx_cab]]
    i_cuenta = cab.index("cuenta")
    i_nif = cab.index("nif")
    i_contra = cab.index("contrapartida") if "contrapartida" in cab else None

    registros = []
    for fila in filas[idx_cab + 1:]:
        def celda(i):
            return fila[i].strip() if i is not None and i < len(fila) else ""
        cuenta = re.sub(r"\.0$", "", celda(i_cuenta))
        nif = limpiar_nif(celda(i_nif))
        if not cuenta.isdigit() or not nif:
            continue  # salta guiones, cuentas de grupo sin NIF, líneas vacías...
        contra = re.sub(r"\.0$", "", celda(i_contra))
        registros.append({"NIF": nif, "Cuenta": cuenta, "Contrapartida": contra})

    df = pd.DataFrame(registros, columns=["NIF", "Cuenta", "Contrapartida"])
    repetidos = sorted(df.loc[df["NIF"].duplicated(keep=False), "NIF"].unique().tolist())
    df = df.drop_duplicates("NIF", keep="first")
    return df, repetidos


def _num(v):
    """Convierte a número (int si es entero) o None."""
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def _txt(v):
    if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s else None


def generar_excel(df_final):
    """Rellena la pestaña Compras de la plantilla y devuelve los bytes del .xlsx."""
    wb = openpyxl.load_workbook(RUTA_PLANTILLA)
    ws = wb[HOJA_COMPRAS]

    for i, fila in enumerate(df_final.to_dict("records")):
        r = FILA_INICIO_COMPRAS + i
        if r > FILA_MODELO_ESTILO:  # más facturas que filas formateadas: copiar estilo
            for c in range(1, 11):
                ws.cell(r, c)._style = copy(ws.cell(FILA_MODELO_ESTILO, c)._style)

        ws.cell(r, 1).value = _txt(fila["FECHA"])
        ws.cell(r, 2).value = _num(fila["COD.PROVEED"])
        ws.cell(r, 3).value = _txt(fila["Nº FRA."])
        ws.cell(r, 4).value = _txt(fila["COMENTARIO"])
        ws.cell(r, 5).value = _num(fila["TOTAL FRA."])
        ws.cell(r, 6).value = _num(fila["COD. GASTOS"])
        ws.cell(r, 7).value = _num(fila["TIPO IVA"])
        ws.cell(r, 8).value = None
        ws.cell(r, 9).value = _num(fila["BASE IMP."])    # None => se borra la fórmula (varios IVA)
        ws.cell(r, 10).value = _num(fila["CUOTA IVA"])
        for c in (2, 6):  # códigos como número, no como texto
            ws.cell(r, c).number_format = "General"

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📄 Asesoría AI - Procesador de Facturas")

with st.expander("ℹ️ ¿Qué tengo que subir? (guía rápida)", expanded=True):
    st.markdown(
        """
**Necesitas 3 cosas:**

**1. Último número interno registrado** (campo numérico)
El último número de registro interno que ya habéis usado. La primera factura de este lote
será ese número **+ 1** y las siguientes irán correlativas.

**2. Informe Plan Contable (CSV o Excel)**
Súbelo **tal cual sale del programa de contabilidad** (*INFORME PLAN CONTABLE*), **sin editarlo**.
La app ignora sola las primeras filas, la línea de guiones y las cuentas sin NIF. Solo usa estas columnas:

| Columna | Para qué se usa |
|---|---|
| `NIF` | Para identificar al proveedor de cada factura |
| `Cuenta` | Sale como **COD.PROVEED** |
| `Contrapartida` | Sale como **COD. GASTOS** |

Si un proveedor no está en el informe, sus códigos salen **vacíos** (te avisamos al terminar).
Si un NIF aparece en dos cuentas, se usa la primera.

**3. Facturas en PDF**
Sube todos los PDFs del lote. **Cada PDF = una factura** (aunque tenga varias páginas).
No metas dos facturas distintas en el mismo PDF.

**Qué obtienes:** el Excel de *entrada de apuntes* con sus 4 pestañas (Ventas, Compras, Estandar y Banco 1).
**Solo se rellena Compras**; las demás quedan vacías.

⚠️ **Facturas con varios tipos de IVA** (ej. 21 % y 10 %): FECHA, TIPO IVA, BASE IMP. y CUOTA IVA
salen **vacíos a propósito** y hay que rellenarlos a mano. Sí salen el nº de factura, el proveedor y el total.
        """
    )

if not os.path.exists(RUTA_PLANTILLA):
    st.error(
        "Falta el archivo **plantilla_entrada_apuntes.xlsx** junto a app.py. "
        "Súbelo al repositorio (misma carpeta que app.py)."
    )

# BARRA LATERAL
with st.sidebar:
    st.header("Configuración")
    modelo_label = st.selectbox("Modelo AI", list(MODELOS.keys()))
    modelo = MODELOS[modelo_label]
    st.info(
        "Flash-Lite es el más barato y suele bastar para facturas normales. "
        "Si falla con facturas complejas o escaneadas, prueba Gemini 3.8 Flash."
    )

# ZONA PRINCIPAL
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
        "2️⃣ Informe Plan Contable (CSV o Excel, tal cual sale del programa)",
        type=["csv", "xlsx", "xls"],
    )

with col2:
    uploaded_files = st.file_uploader(
        "3️⃣ Facturas (PDF, una factura por archivo)",
        type=["pdf"],
        accept_multiple_files=True,
    )

if st.button("🚀 Procesar Facturas", type="primary"):
    if not uploaded_files:
        st.warning("⚠️ No has subido ningún archivo.")
    elif client is None:
        st.error("Falta la API KEY, no se puede procesar.")
    elif not os.path.exists(RUTA_PLANTILLA):
        st.error("Falta la plantilla del Excel (plantilla_entrada_apuntes.xlsx).")
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

            df["NIF"] = df["nif_emisor"].map(limpiar_nif) if "nif_emisor" in df.columns else ""

            for col in ["nombre_proveedor", "fecha", "numero_factura",
                        "base_imponible", "tipo_iva", "cuota_iva", "total_factura"]:
                if col not in df.columns:
                    df[col] = None

            df["COD.PROVEED"] = pd.NA
            df["COD. GASTOS"] = pd.NA

            if archivo_maestro is not None:
                try:
                    df_maestro, repetidos = leer_maestro(archivo_maestro)
                    df = df.drop(columns=["COD.PROVEED", "COD. GASTOS"])
                    df = pd.merge(df, df_maestro, on="NIF", how="left")
                    df = df.rename(columns={"Cuenta": "COD.PROVEED", "Contrapartida": "COD. GASTOS"})
                    if repetidos:
                        st.info("NIF con más de una cuenta en el informe (se usó la primera): " + ", ".join(repetidos))
                except Exception as e:
                    st.error(f"Error al leer el Informe Plan Contable: {e}")

            for col in ["COD.PROVEED", "COD. GASTOS"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            if archivo_maestro is not None:
                sin_cod = df[df["COD.PROVEED"].isna()]
                if not sin_cod.empty:
                    st.warning(
                        f"{len(sin_cod)} factura(s) sin código de proveedor (NIF no encontrado en el informe): "
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

            st.download_button(
                label="📥 Descargar Excel (entrada de apuntes)",
                data=generar_excel(df_final),
                file_name=f"entrada_de_apuntes_desde_{ultimo_numero + 1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("No se pudieron extraer datos de ningún archivo.")
