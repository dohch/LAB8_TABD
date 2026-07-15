#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import time
import sys

def main():
    print("=== EJERCICIO 1: Insertar 30+ Documentos RENACE ===\n")
    
    # 1. Conectar a PostgreSQL (PUERTO 5433)
    print("1. Conectando a PostgreSQL (puerto 5433)...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5433',
            dbname='bdvdb',
            user='bdv',
            password='bdv2026'
        )
        print("   ✅ Conexión exitosa")
    except Exception as e:
        print(f"   ❌ Error al conectar: {e}")
        print("   ⚠️  Verifica que Docker esté corriendo: docker ps")
        print("   ⚠️  Verifica el puerto: docker port pgvector-lab")
        sys.exit(1)
    
    # 2. Registrar el tipo VECTOR (OBLIGATORIO)
    print("2. Registrando tipo VECTOR...")
    register_vector(conn)
    print("   ✅ Tipo VECTOR registrado")
    
    cur = conn.cursor()
    
    # 3. Modelo de embeddings
    print("3. Cargando modelo de embeddings 'all-MiniLM-L6-v2'...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Modelo cargado")
    except Exception as e:
        print(f"   ❌ Error cargando modelo: {e}")
        sys.exit(1)
    
    # 4. Corpus de 30 documentos (10 por categoría)
    print("4. Preparando corpus de 30 documentos...")
    corpus_renace = [
        # --- CATEGORÍA: EDA (10) ---
        {'titulo': 'EDA en menores de 5 años: factores de riesgo en zonas rurales',
         'resumen': 'La EDA afecta principalmente a niños menores de 5 años. Factores de riesgo incluyen agua contaminada, saneamiento deficiente y desnutrición. El zinc reduce duración en 35%.',
         'categoria': 'EDA', 'año': 2024, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Rehidratación oral y lactancia materna en EDA',
         'resumen': 'Sales de rehidratación oral (SRO) son tratamiento de primera línea. Lactancia materna exclusiva protege contra EDA severa. Rotavirus causa 40% de hospitalizaciones.',
         'categoria': 'EDA', 'año': 2023, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Impacto del cambio climático en la incidencia de EDA',
         'resumen': 'El aumento de temperaturas y eventos climáticos extremos incrementan la incidencia de EDA. Se observa una correlación positiva entre inundaciones y brotes.',
         'categoria': 'EDA', 'año': 2024, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Estrategias comunitarias para la prevención de EDA',
         'resumen': 'Programas de lavado de manos y tratamiento de agua a nivel comunitario reducen la incidencia de EDA en un 47%. La participación comunitaria es clave para el éxito.',
         'categoria': 'EDA', 'año': 2023, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Rotavirus: principal causa de EDA grave en Latinoamérica',
         'resumen': 'El rotavirus es responsable de aproximadamente el 40% de las hospitalizaciones por EDA en niños menores de 5 años en la región. La vacunación es la medida más efectiva.',
         'categoria': 'EDA', 'año': 2024, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Suplementación con zinc en el manejo de EDA en niños',
         'resumen': 'La suplementación con zinc reduce la duración y severidad de la EDA. Se recomienda su uso junto con SRO para todos los niños con diarrea aguda.',
         'categoria': 'EDA', 'año': 2022, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'EDA y desnutrición crónica: un círculo vicioso',
         'resumen': 'La desnutrición crónica aumenta la susceptibilidad a EDA, y los episodios recurrentes de EDA empeoran el estado nutricional, creando un círculo vicioso que afecta el desarrollo infantil.',
         'categoria': 'EDA', 'año': 2023, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Vigilancia de EDA en la región amazónica del Perú',
         'resumen': 'La región amazónica presenta una alta incidencia de EDA debido a la falta de acceso a agua potable y saneamiento. La vigilancia epidemiológica es crucial para la detección temprana de brotes.',
         'categoria': 'EDA', 'año': 2024, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Resistencia a antibióticos en bacterias causantes de EDA',
         'resumen': 'Se ha reportado un aumento en la resistencia a antibióticos en cepas de Shigella y Salmonella, lo que complica el manejo de la EDA y requiere un uso más racional de antimicrobianos.',
         'categoria': 'EDA', 'año': 2023, 'fuente': 'RENACE-MINSA'},
        {'titulo': 'Telemedicina para el diagnóstico y seguimiento de EDA',
         'resumen': 'La telemedicina permite el diagnóstico temprano y el seguimiento de casos de EDA en áreas remotas, mejorando el acceso a la atención y reduciendo complicaciones.',
         'categoria': 'EDA', 'año': 2024, 'fuente': 'RENACE-MINSA'},
        
        # --- CATEGORÍA: DENGUE (10) ---
        {'titulo': 'Vigilancia epidemiológica dengue Loreto 2024',
         'resumen': 'Loreto concentra 45% de casos dengue en Perú. Índice aédico supera 5% en zonas de riesgo. Aedes aegypti presente en el 78% de viviendas inspeccionadas.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Control vectorial dengue: eliminación de criaderos',
         'resumen': 'Eliminación de criaderos reduce índice aédico en 60%. Participación comunitaria es clave. Fumigación adulticida sola no es suficiente sin control de criaderos.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Nuevas técnicas de control biológico para el Aedes aegypti',
         'resumen': 'El uso de la bacteria Wolbachia y mosquitos estériles son nuevas técnicas prometedoras para el control del Aedes aegypti, reduciendo la transmisión del dengue.',
         'categoria': 'Dengue', 'año': 2023, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Cambio climático y expansión del dengue en zonas andinas',
         'resumen': 'El cambio climático está permitiendo la expansión del vector Aedes aegypti a zonas andinas previamente libres de dengue, lo que representa un nuevo desafío para la salud pública.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Dengue grave: factores de riesgo y manejo clínico',
         'resumen': 'El dengue grave se asocia con factores como edad avanzada, comorbilidades y coinfección. El manejo clínico oportuno con hidratación y monitoreo es esencial para reducir la mortalidad.',
         'categoria': 'Dengue', 'año': 2023, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Efectividad de la vacuna contra el dengue en Brasil',
         'resumen': 'Estudios en Brasil muestran que la vacuna contra el dengue (Qdenga) tiene una efectividad del 80% en la prevención de casos graves y hospitalizaciones.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Diagnóstico diferencial entre dengue, zika y chikungunya',
         'resumen': 'Los síntomas de dengue, zika y chikungunya son similares, lo que dificulta el diagnóstico clínico. Las pruebas de laboratorio como RT-PCR son fundamentales para un diagnóstico certero.',
         'categoria': 'Dengue', 'año': 2023, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Estrategias de comunicación para la prevención del dengue',
         'resumen': 'Las campañas de comunicación efectivas que utilizan redes sociales y medios locales han logrado aumentar la participación comunitaria en la eliminación de criaderos y reducir la incidencia de dengue.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Modelado matemático para predecir brotes de dengue',
         'resumen': 'El modelado matemático que integra datos climáticos, de vigilancia vectorial y de casos permite predecir con anticipación los brotes de dengue, facilitando la asignación de recursos.',
         'categoria': 'Dengue', 'año': 2023, 'fuente': 'DGE-MINSA'},
        {'titulo': 'Resistencia a insecticidas en poblaciones de Aedes aegypti',
         'resumen': 'Se ha detectado resistencia a insecticidas piretroides en poblaciones de Aedes aegypti en varias regiones del Perú, lo que requiere un cambio en las estrategias de control químico.',
         'categoria': 'Dengue', 'año': 2024, 'fuente': 'DGE-MINSA'},
        
        # --- CATEGORÍA: TUBERCULOSIS (10) ---
        {'titulo': 'TB-MDR en Lima: resistencia a rifampicina',
         'resumen': 'Lima concentra 62% de casos TB-MDR en Perú. GeneXpert detecta resistencia rifampicina en 2 horas. Esquema bedaquilina-linezolid muestra 85% de éxito.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'DOTS y adherencia al tratamiento TB en Arequipa',
         'resumen': 'Estrategia DOTS logra 91% de éxito en tratamiento en Arequipa. Supervisión directa es clave para evitar resistencia. Abandono del tratamiento genera TB-MDR.',
         'categoria': 'Tuberculosis', 'año': 2023, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Impacto de la pandemia COVID-19 en el diagnóstico de TB',
         'resumen': 'La pandemia de COVID-19 interrumpió los servicios de diagnóstico y tratamiento de TB, lo que llevó a una disminución en la detección de casos y un aumento en la mortalidad.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Nuevos fármacos para el tratamiento de TB-MDR',
         'resumen': 'El uso de bedaquilina, delamanid y linezolid ha revolucionado el tratamiento de la TB-MDR, acortando la duración y mejorando la tasa de éxito en comparación con los esquemas antiguos.',
         'categoria': 'Tuberculosis', 'año': 2023, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'TB en poblaciones indígenas: desafíos y estrategias',
         'resumen': 'Las poblaciones indígenas tienen una incidencia de TB desproporcionadamente alta debido a la pobreza, desnutrición y falta de acceso a servicios de salud. Se requieren estrategias culturalmente apropiadas.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Diagnóstico molecular de TB: GeneXpert y más allá',
         'resumen': 'GeneXpert ha mejorado el diagnóstico de TB, pero se necesitan nuevas herramientas como la secuenciación de nueva generación (NGS) para detectar resistencia a fármacos de manera más completa.',
         'categoria': 'Tuberculosis', 'año': 2023, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Estrategias para mejorar la adherencia al tratamiento TB',
         'resumen': 'El uso de recordatorios por mensajes de texto, el apoyo de pares y la entrega de alimentos han demostrado ser efectivos para mejorar la adherencia al tratamiento de TB, especialmente en poblaciones vulnerables.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Coinfección TB-VIH: un desafío creciente',
         'resumen': 'La coinfección TB-VIH es un problema de salud pública importante. El manejo integrado de ambas infecciones es crucial para reducir la mortalidad y mejorar la calidad de vida de los pacientes.',
         'categoria': 'Tuberculosis', 'año': 2023, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'TB latente: diagnóstico y tratamiento en contactos',
         'resumen': 'El diagnóstico y tratamiento de la infección latente por TB en contactos de casos índice es una estrategia clave para la eliminación de la TB. La prueba de IGRA es la más específica.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'},
        {'titulo': 'Uso de inteligencia artificial para el diagnóstico de TB',
         'resumen': 'Algoritmos de IA aplicados a radiografías de tórax pueden ayudar en el diagnóstico temprano de TB, especialmente en áreas con escasez de radiólogos, mejorando la tasa de detección.',
         'categoria': 'Tuberculosis', 'año': 2024, 'fuente': 'PCTB-MINSA'}
    ]
    
    print(f"   ✅ {len(corpus_renace)} documentos preparados")
    
    # 5. Generar embeddings
    print("5. Generando embeddings (esto puede tomar unos segundos)...")
    textos = [f"{d['titulo']}. {d['resumen']}" for d in corpus_renace]
    start_time = time.time()
    embs = model.encode(textos, normalize_embeddings=True).astype(np.float32)
    end_time = time.time()
    print(f"   ✅ {len(embs)} embeddings generados en {end_time - start_time:.2f} segundos")
    print(f"   📐 Dimensión de cada vector: {embs.shape[1]}")
    
    # 6. Insertar datos
    print("6. Insertando documentos en la base de datos...")
    
    # Limpiar tabla por si tiene datos previos
    cur.execute("TRUNCATE TABLE articulos RESTART IDENTITY;")
    print("   ✅ Tabla 'articulos' limpiada")
    
    # Preparar consulta SQL
    insert_sql = """
    INSERT INTO articulos (titulo, resumen, categoria, año, fuente, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    inserted = 0
    for i, doc in enumerate(corpus_renace):
        try:
            cur.execute(insert_sql, (
                doc['titulo'],
                doc['resumen'],
                doc['categoria'],
                doc['año'],
                doc['fuente'],
                embs[i]
            ))
            inserted += 1
            if (i + 1) % 10 == 0:
                print(f"   📝 Insertados {i + 1} documentos...")
        except Exception as e:
            print(f"   ❌ Error en documento {i+1}: {e}")
    
    conn.commit()
    print(f"   ✅ {inserted} documentos insertados correctamente")
    
    # 7. Verificar inserción
    print("7. Verificando inserción...")
    cur.execute("SELECT COUNT(*) FROM articulos;")
    total = cur.fetchone()[0]
    print(f"   ✅ Total de registros en 'articulos': {total}")
    
    # 8. Mostrar muestra
    print("\n8. Muestra de datos insertados:")
    cur.execute("""
        SELECT id, titulo, categoria, año 
        FROM articulos 
        ORDER BY id 
        LIMIT 5;
    """)
    print("   ID | Título (primeros 40 chars) | Categoría | Año")
    print("   " + "-" * 75)
    for row in cur.fetchall():
        titulo = row[1][:40] + "..." if len(row[1]) > 40 else row[1]
        print(f"   {row[0]:2} | {titulo:40} | {row[2]:12} | {row[3]:4}")
    
    cur.close()
    conn.close()
    print("\n✅ ¡EJERCICIO 1 COMPLETADO!")
    print("   Conexión a PostgreSQL cerrada.")

if __name__ == "__main__":
    main()