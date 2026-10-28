#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesamiento en batch con Apache Spark - Resultados Saber Pro
Curso Big Data (202016911) - UNAD - Tarea 3

Problema: ¿Cómo se relacionan el estrato socioeconómico, el método del programa
(presencial / distancia), el género y el departamento de residencia con los
puntajes obtenidos en las pruebas Saber Pro?

Flujo:
  1. Carga del CSV original (todas las columnas como texto).
  2. Limpieza y transformación -> se guarda en formato Parquet (una sola lectura
     del CSV de ~800 MB; las demás consultas leen el Parquet, mucho más liviano).
  3. Calidad de datos y análisis exploratorio (EDA) con DataFrames
     (y una demostración corta con RDDs: map + reduceByKey).
  4. Resultados en CSV y gráficas PNG.

Uso:
  spark-submit --driver-memory 1g batch_saber_pro.py [ruta_csv] [carpeta_salida]
  (por defecto: ~/saber_pro.csv  y  ~/resultados_batch)
"""
import os
import sys
from operator import add

import matplotlib
matplotlib.use("Agg")  # sin entorno gráfico (la VM solo tiene terminal)
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------
HOME = os.path.expanduser("~")
RUTA_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "saber_pro.csv")
SALIDA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "resultados_batch")
RUTA_PARQUET = os.path.join(SALIDA, "saber_pro_limpio.parquet")
RUTA_GRAFICAS = os.path.join(SALIDA, "graficas")
RUTA_TABLAS = os.path.join(SALIDA, "tablas")
for carpeta in (SALIDA, RUTA_GRAFICAS, RUTA_TABLAS):
    os.makedirs(carpeta, exist_ok=True)

MODULOS = {
    "MOD_RAZONA_CUANTITAT_PUNT": "razonamiento_cuantitativo",
    "MOD_LECTURA_CRITICA_PUNT": "lectura_critica",
    "MOD_COMPETEN_CIUDADA_PUNT": "competencias_ciudadanas",
    "MOD_COMUNI_ESCRITA_PUNT": "comunicacion_escrita",
    "MOD_INGLES_PUNT": "ingles",
}
COLS_TEXTO = {
    "ESTU_CONSECUTIVO": "consecutivo",
    "PERIODO": "periodo",
    "ESTU_GENERO": "genero",
    "FAMI_ESTRATOVIVIENDA": "estrato_txt",
    "ESTU_METODO_PRGM": "metodo",
    "ESTU_PAIS_RESIDE": "pais",
    "ESTU_DEPTO_RESIDE": "departamento",
    "INST_CARACTER_ACADEMICO": "caracter_inst",
    "INST_ORIGEN": "origen_inst",
    "ESTU_HORASSEMANATRABAJA": "horas_trabajo",
    "FAMI_TIENEINTERNET": "tiene_internet",
    "FAMI_TIENECOMPUTADOR": "tiene_computador",
}
NOMBRES_MODULOS = list(MODULOS.values())
ORDEN_MODULOS_LEGIBLE = {
    "razonamiento_cuantitativo": "Razonamiento cuantitativo",
    "lectura_critica": "Lectura crítica",
    "competencias_ciudadanas": "Competencias ciudadanas",
    "comunicacion_escrita": "Comunicación escrita",
    "ingles": "Inglés",
}


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def a_pandas(df):
    """Convierte un DataFrame pequeño de Spark a pandas sin depender de Arrow."""
    filas = [r.asDict() for r in df.collect()]
    return pd.DataFrame(filas)


def guardar_tabla(df_pandas, nombre):
    df_pandas.to_csv(os.path.join(RUTA_TABLAS, nombre + ".csv"), index=False, encoding="utf-8-sig")


def guardar_grafica(nombre):
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICAS, nombre + ".png"), dpi=130)
    plt.close()


def etiquetas_barras(ax, fmt="{:.1f}", horizontal=False):
    for p in ax.patches:
        if horizontal:
            ax.annotate(fmt.format(p.get_width()), (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha="left", va="center", fontsize=8, xytext=(3, 0), textcoords="offset points")
        else:
            ax.annotate(fmt.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


spark = (SparkSession.builder
         .appName("BatchSaberPro")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ----------------------------------------------------------------------------
# 1. CARGA + LIMPIEZA (una sola lectura del CSV original)
# ----------------------------------------------------------------------------
titulo("1. CARGA Y LIMPIEZA DEL CSV ORIGINAL")
print("Archivo de entrada:", RUTA_CSV)

crudo = (spark.read
         .option("header", True)
         .option("quote", '"')
         .option("escape", '"')
         .option("encoding", "UTF-8")
         .csv("file://" + RUTA_CSV))   # todo como texto: sin inferSchema (evita una pasada extra)

print("Columnas del archivo original:", len(crudo.columns))


def limpiar_texto(nombre_col):
    """Quita espacios y convierte cadenas vacías en nulos."""
    c = F.trim(F.col(nombre_col))
    return F.when(c == "", None).otherwise(c)


seleccion = [limpiar_texto(orig).alias(nuevo) for orig, nuevo in COLS_TEXTO.items()]
for orig, nuevo in MODULOS.items():
    puntaje = limpiar_texto(orig).cast("double")
    # Un puntaje válido está entre 0 y 300; lo demás se trata como dato faltante.
    seleccion.append(F.when((puntaje >= 0) & (puntaje <= 300), puntaje).alias(nuevo))

limpio = crudo.select(*seleccion)

# Transformaciones: estrato numérico, método normalizado, promedio de módulos
limpio = (limpio
          .withColumn("estrato", F.nullif(F.regexp_extract("estrato_txt", r"(\d)", 1), F.lit("")).cast("int"))
          .withColumn("metodo", F.when(F.col("metodo") == "DISTANCIA VITUAL", "DISTANCIA VIRTUAL")
                      .otherwise(F.col("metodo")))
          .withColumn("departamento", F.upper(F.col("departamento")))
          .withColumn("promedio_modulos",
                      F.when(F.expr(" AND ".join(f"{m} IS NOT NULL" for m in NOMBRES_MODULOS)),
                             sum(F.col(m) for m in NOMBRES_MODULOS) / len(NOMBRES_MODULOS))))

print("Escribiendo datos limpios en Parquet (esta es la única lectura pesada)...")
limpio.write.mode("overwrite").parquet("file://" + RUTA_PARQUET)

# ----------------------------------------------------------------------------
# 2. CALIDAD DE DATOS (se calcula sobre el Parquet, ya liviano)
# ----------------------------------------------------------------------------
df = spark.read.parquet("file://" + RUTA_PARQUET)

titulo("2. CALIDAD DE DATOS")
total_bruto = df.count()
df = df.dropDuplicates(["consecutivo"])      # un registro por estudiante
df.persist()                                  # se reutiliza en todas las consultas
total = df.count()
print(f"Filas leídas del CSV:              {total_bruto:,}")
print(f"Duplicados eliminados (consecutivo): {total_bruto - total:,}")
print(f"Filas para el análisis:             {total:,}")

nulos = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
                   for c in ["genero", "estrato", "metodo", "departamento", "horas_trabajo"] + NOMBRES_MODULOS])
nulos_pd = a_pandas(nulos).T.reset_index()
nulos_pd.columns = ["columna", "nulos"]
nulos_pd["porcentaje"] = (nulos_pd["nulos"] / total * 100).round(2)
print("\nValores faltantes por columna:")
print(nulos_pd.to_string(index=False))
guardar_tabla(nulos_pd, "00_calidad_nulos")

completos = df.filter(F.col("promedio_modulos").isNotNull()).count()
print(f"\nEstudiantes con los 5 módulos completos: {completos:,} ({completos / total * 100:.1f} %)")

# ----------------------------------------------------------------------------
# 3. EDA CON DATAFRAMES
# ----------------------------------------------------------------------------
titulo("3.1 ESTADÍSTICAS DESCRIPTIVAS POR MÓDULO")
descr = df.select(
    *[F.round(F.mean(m), 1).alias(f"{m}__media") for m in NOMBRES_MODULOS],
    *[F.round(F.stddev(m), 1).alias(f"{m}__desv") for m in NOMBRES_MODULOS],
    *[F.min(m).alias(f"{m}__min") for m in NOMBRES_MODULOS],
    *[F.max(m).alias(f"{m}__max") for m in NOMBRES_MODULOS],
).first().asDict()
descr_pd = pd.DataFrame([
    {"modulo": ORDEN_MODULOS_LEGIBLE[m], "media": descr[f"{m}__media"], "desv_est": descr[f"{m}__desv"],
     "minimo": descr[f"{m}__min"], "maximo": descr[f"{m}__max"]} for m in NOMBRES_MODULOS])
print(descr_pd.to_string(index=False))
guardar_tabla(descr_pd, "01_descriptivas_modulos")

titulo("3.2 PROMEDIO DE MÓDULOS POR ESTRATO")
estrato = (df.filter(F.col("estrato").isNotNull() & F.col("promedio_modulos").isNotNull())
           .groupBy("estrato")
           .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
           .orderBy("estrato"))
estrato.show()
estrato_pd = a_pandas(estrato)
guardar_tabla(estrato_pd, "02_promedio_por_estrato")

titulo("3.3 PROMEDIO POR MÉTODO DEL PROGRAMA")
metodo = (df.filter(F.col("metodo").isNotNull() & F.col("promedio_modulos").isNotNull())
          .groupBy("metodo")
          .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
          .orderBy(F.desc("promedio")))
metodo.show(truncate=False)
metodo_pd = a_pandas(metodo)
guardar_tabla(metodo_pd, "03_promedio_por_metodo")

titulo("3.4 PROMEDIO POR MÓDULO Y GÉNERO")
genero = (df.filter(F.col("genero").isin("F", "M"))
          .groupBy("genero")
          .agg(F.count("*").alias("estudiantes"),
               *[F.round(F.avg(m), 1).alias(m) for m in NOMBRES_MODULOS]))
genero.show(truncate=False)
genero_pd = a_pandas(genero).set_index("genero")
guardar_tabla(genero_pd.reset_index(), "04_promedio_por_genero")

titulo("3.5 PROMEDIO POR MÓDULO Y ESTRATO")
mod_estrato = (df.filter(F.col("estrato").isNotNull())
               .groupBy("estrato")
               .agg(*[F.round(F.avg(m), 1).alias(m) for m in NOMBRES_MODULOS])
               .orderBy("estrato"))
mod_estrato.show(truncate=False)
mod_estrato_pd = a_pandas(mod_estrato)
guardar_tabla(mod_estrato_pd, "05_modulos_por_estrato")

titulo("3.6 DEPARTAMENTOS (residencia en Colombia, mínimo 2.000 estudiantes)")
depto = (df.filter((F.col("pais") == "COLOMBIA") & F.col("departamento").isNotNull()
                   & F.col("promedio_modulos").isNotNull())
         .groupBy("departamento")
         .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
         .filter(F.col("estudiantes") >= 2000)
         .orderBy(F.desc("promedio")))
depto.show(40, truncate=False)
depto_pd = a_pandas(depto)
guardar_tabla(depto_pd, "06_promedio_por_departamento")

titulo("3.7 PROMEDIO POR ORIGEN DE LA INSTITUCIÓN Y HORAS DE TRABAJO")
origen = (df.filter(F.col("origen_inst").isNotNull() & F.col("promedio_modulos").isNotNull())
          .groupBy("origen_inst")
          .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
          .filter(F.col("estudiantes") >= 1000)
          .orderBy(F.desc("promedio")))
origen.show(truncate=False)
guardar_tabla(a_pandas(origen), "07_promedio_por_origen_institucion")

horas = (df.filter(F.col("horas_trabajo").isNotNull() & F.col("promedio_modulos").isNotNull())
         .groupBy("horas_trabajo")
         .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
         .orderBy(F.desc("promedio")))
horas.show(truncate=False)
guardar_tabla(a_pandas(horas), "08_promedio_por_horas_trabajo")

titulo("3.8 ACCESO A INTERNET Y COMPUTADOR")
acceso = (df.filter(F.col("tiene_internet").isin("Si", "No") & F.col("tiene_computador").isin("Si", "No")
                    & F.col("promedio_modulos").isNotNull())
          .groupBy("tiene_internet", "tiene_computador")
          .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
          .orderBy("tiene_internet", "tiene_computador"))
acceso.show()
guardar_tabla(a_pandas(acceso), "09_acceso_internet_computador")

titulo("3.9 EVOLUCIÓN POR PERIODO (periodos con más de 5.000 estudiantes)")
periodo = (df.filter(F.col("promedio_modulos").isNotNull() & F.col("periodo").isNotNull())
           .groupBy("periodo")
           .agg(F.count("*").alias("estudiantes"), F.round(F.avg("promedio_modulos"), 1).alias("promedio"))
           .filter(F.col("estudiantes") > 5000)
           .orderBy("periodo"))
periodo.show()
periodo_pd = a_pandas(periodo)
guardar_tabla(periodo_pd, "10_promedio_por_periodo")

titulo("3.10 DISTRIBUCIÓN DEL PROMEDIO DE MÓDULOS (rangos de 20 puntos)")
dist = (df.filter(F.col("promedio_modulos").isNotNull())
        .withColumn("rango", (F.floor(F.col("promedio_modulos") / 20) * 20).cast("int"))
        .groupBy("rango").agg(F.count("*").alias("estudiantes")).orderBy("rango"))
dist.show(20)
dist_pd = a_pandas(dist)
guardar_tabla(dist_pd, "11_distribucion_promedio")

titulo("3.11 CORRELACIÓN ENTRE MÓDULOS")
corr = pd.DataFrame(index=[ORDEN_MODULOS_LEGIBLE[m] for m in NOMBRES_MODULOS],
                    columns=[ORDEN_MODULOS_LEGIBLE[m] for m in NOMBRES_MODULOS], dtype=float)
completos_df = df.filter(F.col("promedio_modulos").isNotNull())
for a in NOMBRES_MODULOS:
    for b in NOMBRES_MODULOS:
        corr.loc[ORDEN_MODULOS_LEGIBLE[a], ORDEN_MODULOS_LEGIBLE[b]] = 1.0 if a == b else completos_df.stat.corr(a, b)
print(corr.round(2).to_string())
corr.round(3).to_csv(os.path.join(RUTA_TABLAS, "12_correlacion_modulos.csv"), encoding="utf-8-sig")

# ----------------------------------------------------------------------------
# 4. DEMOSTRACIÓN CON RDDs  (map + reduceByKey + takeOrdered)
# ----------------------------------------------------------------------------
titulo("4. DEMOSTRACIÓN CON RDDs: estudiantes por departamento")
rdd = df.select("departamento").rdd                      # DataFrame -> RDD de filas
conteo = (rdd.map(lambda fila: (fila["departamento"], 1))   # transformación: map
             .filter(lambda par: par[0] is not None)        # transformación: filter
             .reduceByKey(add))                             # transformación: reduceByKey
print("Total de departamentos distintos (acción count):", conteo.count())
print("Top 5 departamentos por número de estudiantes (acción takeOrdered):")
for nombre, n in conteo.takeOrdered(5, key=lambda par: -par[1]):
    print(f"  {nombre:<20} {n:>10,}")

# ----------------------------------------------------------------------------
# 5. GRÁFICAS
# ----------------------------------------------------------------------------
titulo("5. GENERANDO GRÁFICAS")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
AZUL, NARANJA, VERDE = "#1f77b4", "#ff7f0e", "#2ca02c"

# 5.1 Estrato
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(estrato_pd["estrato"].astype(str), estrato_pd["promedio"], color=AZUL)
etiquetas_barras(ax)
ax.set_title("Promedio de módulos Saber Pro según el estrato")
ax.set_xlabel("Estrato de la vivienda")
ax.set_ylabel("Puntaje promedio (0-300)")
guardar_grafica("01_promedio_por_estrato")

# 5.2 Método
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(metodo_pd["metodo"], metodo_pd["promedio"], color=[VERDE, NARANJA, AZUL][:len(metodo_pd)])
etiquetas_barras(ax)
ax.set_title("Promedio de módulos según el método del programa")
ax.set_ylabel("Puntaje promedio (0-300)")
guardar_grafica("02_promedio_por_metodo")

# 5.3 Módulos por estrato (líneas)
fig, ax = plt.subplots(figsize=(8, 4.8))
for m in NOMBRES_MODULOS:
    ax.plot(mod_estrato_pd["estrato"], mod_estrato_pd[m], marker="o", label=ORDEN_MODULOS_LEGIBLE[m])
ax.set_title("Puntaje promedio por módulo y estrato")
ax.set_xlabel("Estrato de la vivienda")
ax.set_ylabel("Puntaje promedio (0-300)")
ax.legend(fontsize=8)
guardar_grafica("03_modulos_por_estrato")

# 5.4 Género por módulo
fig, ax = plt.subplots(figsize=(9, 4.8))
x = range(len(NOMBRES_MODULOS))
ancho = 0.38
if "F" in genero_pd.index and "M" in genero_pd.index:
    ax.bar([i - ancho / 2 for i in x], [genero_pd.loc["F", m] for m in NOMBRES_MODULOS], ancho,
           label="Femenino", color=NARANJA)
    ax.bar([i + ancho / 2 for i in x], [genero_pd.loc["M", m] for m in NOMBRES_MODULOS], ancho,
           label="Masculino", color=AZUL)
ax.set_xticks(list(x))
ax.set_xticklabels([ORDEN_MODULOS_LEGIBLE[m].replace(" ", "\n", 1) for m in NOMBRES_MODULOS], fontsize=8)
ax.set_title("Puntaje promedio por módulo y género")
ax.set_ylabel("Puntaje promedio (0-300)")
ax.legend()
guardar_grafica("04_modulos_por_genero")

# 5.5 Departamentos (mejores 10 y últimos 5)
if len(depto_pd) > 0:
    sel = pd.concat([depto_pd.head(10), depto_pd.tail(5)]).drop_duplicates("departamento")
    sel = sel.sort_values("promedio")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sel["departamento"], sel["promedio"], color=AZUL)
    etiquetas_barras(ax, horizontal=True)
    ax.set_title("Promedio de módulos por departamento de residencia\n(10 más altos y 5 más bajos, mín. 2.000 estudiantes)")
    ax.set_xlabel("Puntaje promedio (0-300)")
    guardar_grafica("05_promedio_por_departamento")

# 5.6 Distribución
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(dist_pd["rango"] + 10, dist_pd["estudiantes"], width=18, color=VERDE)
ax.set_title("Distribución del promedio de módulos (rangos de 20 puntos)")
ax.set_xlabel("Promedio de los 5 módulos")
ax.set_ylabel("Número de estudiantes")
guardar_grafica("06_distribucion_promedio")

# 5.7 Periodo
if len(periodo_pd) > 0:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(periodo_pd["periodo"], periodo_pd["promedio"], marker="o", color=AZUL)
    for _, fila in periodo_pd.iterrows():
        ax.annotate(f"{fila['promedio']:.1f}", (fila["periodo"], fila["promedio"]),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 5), textcoords="offset points")
    ax.set_title("Promedio de módulos por periodo de aplicación")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Puntaje promedio (0-300)")
    guardar_grafica("07_promedio_por_periodo")

print("Gráficas guardadas en:", RUTA_GRAFICAS)
print("Tablas CSV guardadas en:", RUTA_TABLAS)
titulo("PROCESAMIENTO BATCH FINALIZADO")
spark.stop()
