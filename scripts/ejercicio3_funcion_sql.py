#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import pandas as pd

def main():
    print("=" * 80)
    print("EJERCICIO 3: Función SQL de Búsqueda Semántica")
    print("=" * 80)
    
    # 1. Conectar a PostgreSQL
    print("\n1. Conectando a PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5433',
            dbname='bdvdb',
            user='bdv',
            password='bdv2026'
        )
        register_vector(conn)
        print("   ✅ Conexión exitosa")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    cur = conn.cursor()
    
    # 2. Verificar datos
    cur.execute("SELECT COUNT(*) FROM articulos;")
    total = cur.fetchone()[0]
    print(f"\n2. Verificando datos...")
    print(f"   📊 Documentos en la BD: {total}")
    
    if total < 30:
        print("   ⚠️  ¡ALERTA! Ejecuta primero el Ejercicio 1")
        cur.close()
        conn.close()
        return
    print("   ✅ Base de datos lista")
    
    # 3. Crear la función en PostgreSQL (si no existe)
    print("\n3. Creando función 'buscar_semantico' en PostgreSQL...")
    
    # Eliminar función si existe
    cur.execute("DROP FUNCTION IF EXISTS buscar_semantico(VECTOR(384), INTEGER, VARCHAR);")
    
    # Crear función
    cur.execute("""
        CREATE OR REPLACE FUNCTION buscar_semantico(
            query_vec VECTOR(384),
            k INTEGER DEFAULT 5,
            cat_filtro VARCHAR DEFAULT NULL
        )
        RETURNS TABLE (
            id INTEGER,
            titulo TEXT,
            categoria VARCHAR,
            año INTEGER,
            similitud FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                a.id,
                a.titulo,
                a.categoria,
                a.año,
                1 - (a.embedding <=> query_vec) AS similitud
            FROM articulos a
            WHERE cat_filtro IS NULL OR a.categoria = cat_filtro
            ORDER BY a.embedding <=> query_vec
            LIMIT k;
        END;
        $$ LANGUAGE plpgsql STABLE;
    """)
    conn.commit()
    print("   ✅ Función 'buscar_semantico' creada exitosamente")
    
    # 4. Cargar modelo de embeddings
    print("\n4. Cargando modelo de embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Modelo cargado")
    
    # 5. Definir queries de prueba
    queries_prueba = [
        {"query": "tratamiento de tuberculosis con nuevos medicamentos", "filtro": None},
        {"query": "factores de riesgo en enfermedades diarreicas", "filtro": "EDA"},
        {"query": "control del mosquito Aedes aegypti", "filtro": "Dengue"},
        {"query": "resistencia a antibióticos", "filtro": None},
    ]
    
    print(f"\n5. Preparando {len(queries_prueba)} queries de prueba...")
    
    # 6. Ejecutar pruebas
    print("\n6. Ejecutando búsquedas semánticas con la función SQL...")
    
    resultados_totales = []
    
    for i, test in enumerate(queries_prueba, 1):
        query_texto = test["query"]
        filtro = test["filtro"]
        
        print(f"\n   {'='*70}")
        print(f"   Prueba {i}: '{query_texto}'")
        if filtro:
            print(f"   Filtro: categoría = '{filtro}'")
        else:
            print(f"   Filtro: Ninguno (todos los documentos)")
        print(f"   {'-'*70}")
        
        # Generar embedding
        q_vec = model.encode([query_texto], normalize_embeddings=True)[0].astype(np.float32)
        
        # Llamar a la función SQL desde Python
        try:
            # Usar la función con los parámetros
            sql = """
                SELECT * FROM buscar_semantico(
                    CAST(%s AS vector),
                    %s,
                    %s
                );
            """
            cur.execute(sql, (q_vec.tolist(), 5, filtro))
            resultados = cur.fetchall()
            
            # Mostrar resultados
            print(f"   {'ID':<4} {'Título (50 chars)':<50} {'Categoría':<12} {'Año':<6} {'Similitud':<10}")
            print(f"   {'-'*82}")
            for row in resultados:
                id_, titulo, categoria, año, similitud = row
                titulo_corto = titulo[:50] + "..." if len(titulo) > 50 else titulo
                print(f"   {id_:<4} {titulo_corto:<50} {categoria:<12} {año:<6} {similitud:.4f}")
            
            # Guardar para análisis
            resultados_totales.append({
                "Query": query_texto[:30] + "..." if len(query_texto) > 30 else query_texto,
                "Filtro": filtro if filtro else "Ninguno",
                "Resultados": len(resultados),
                "Top 1": resultados[0][1][:40] + "..." if resultados and len(resultados[0][1]) > 40 else resultados[0][1] if resultados else "N/A",
                "Top 1 Sim": resultados[0][4] if resultados else 0
            })
            
        except Exception as e:
            print(f"   ❌ Error ejecutando búsqueda: {e}")
    
    # 7. Verificar la función en PostgreSQL
    print("\n" + "=" * 80)
    print("7. Verificando la función en PostgreSQL...")
    print("=" * 80)
    
    cur.execute("""
        SELECT 
            proname AS nombre_funcion,
            pg_get_functiondef(oid) AS definicion
        FROM pg_proc
        WHERE proname = 'buscar_semantico';
    """)
    func_info = cur.fetchone()
    if func_info:
        print(f"\n   ✅ Función encontrada: {func_info[0]}")
        print("\n   📝 Definición de la función:")
        print("   " + "-" * 70)
        # Mostrar solo primeras líneas de la definición
        definicion = func_info[1]
        lineas = definicion.split('\n')[:10]
        for linea in lineas:
            print(f"   {linea}")
        if len(definicion.split('\n')) > 10:
            print("   ...")
    else:
        print("   ❌ Función no encontrada")
    
    # 8. Mostrar tabla resumen
    print("\n" + "=" * 80)
    print("8. RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    if resultados_totales:
        df = pd.DataFrame(resultados_totales)
        print(df.to_string(index=False))
    
    # 9. Análisis
    print("\n" + "=" * 80)
    print("9. ANÁLISIS Y CONCLUSIONES")
    print("=" * 80)
    
    print("""
    📌 CONCLUSIONES DEL EJERCICIO 3:
    
    1. La función SQL 'buscar_semantico' encapsula la búsqueda vectorial
       permitiendo reutilización y parametrización.
    
    2. Ventajas de usar funciones SQL:
       - Código reutilizable en múltiples aplicaciones
       - Optimización por el planificador de PostgreSQL
       - Seguridad (evita inyección SQL)
       - Aislación de la lógica de búsqueda
    
    3. Los parámetros permiten flexibilidad:
       - query_vec: vector de búsqueda (obligatorio)
       - k: número de resultados (default 5)
       - cat_filtro: filtro por categoría (opcional)
    
    4. La búsqueda híbrida (vector + filtro) se realiza en una sola
       consulta SQL, demostrando el poder de pgvector.
    
    5. La función es STABLE, lo que permite optimizaciones de caché
       en PostgreSQL.
    """)
    
    # 10. Limpiar
    cur.close()
    conn.close()
    print("\n✅ ¡EJERCICIO 3 COMPLETADO!")
    print("   Conexión a PostgreSQL cerrada.")

if __name__ == "__main__":
    main()