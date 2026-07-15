#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 80)
    print("EJERCICIO 4: Full-Text Search + Búsqueda Vectorial")
    print("=" * 80)
    
    # Conectar a PostgreSQL
    print("\n1. Conectando a PostgreSQL...")
    conn = psycopg2.connect(
        host='localhost',
        port='5433',
        dbname='bdvdb',
        user='bdv',
        password='bdv2026'
    )
    register_vector(conn)
    cur = conn.cursor()
    print("   ✅ Conexión exitosa")
    
    # Verificar datos
    cur.execute("SELECT COUNT(*) FROM articulos;")
    total = cur.fetchone()[0]
    print(f"\n2. Datos disponibles: {total} documentos")
    
    # Verificar columna ts_doc
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'articulos' 
        AND column_name = 'ts_doc';
    """)
    if cur.fetchone():
        print("   ✅ Columna ts_doc existe")
    else:
        print("   ❌ Error: Columna ts_doc no encontrada. Ejecuta primero el comando SQL.")
        return
    
    # Cargar modelo de embeddings
    print("\n3. Cargando modelo de embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Modelo cargado")
    
    # Queries de prueba
    queries = [
        "rotavirus en niños",
        "tratamiento de tuberculosis",
        "control de mosquitos"
    ]
    
    print("\n4. Ejecutando búsquedas híbridas...")
    print("-" * 80)
    
    resultados_totales = []
    
    for query_texto in queries:
        print(f"\n🔍 QUERY: '{query_texto}'")
        print("-" * 70)
        
        # Generar embedding vectorial
        q_vec = model.encode([query_texto], normalize_embeddings=True)[0].astype(np.float32)
        
        # 1. Búsqueda por texto completo (Full-Text Search)
        sql_fts = """
            SELECT id, titulo, ts_rank(ts_doc, to_tsquery('spanish', %s)) AS relevancia
            FROM articulos
            WHERE ts_doc @@ to_tsquery('spanish', %s)
            ORDER BY relevancia DESC
            LIMIT 3;
        """
        # Convertir query a formato tsquery (ej: "rotavirus & niños")
        ts_query = query_texto.replace(' ', ' & ')
        
        print("\n   📝 RESULTADOS POR TEXTO COMPLETO:")
        cur.execute(sql_fts, (ts_query, ts_query))
        fts_resultados = cur.fetchall()
        if fts_resultados:
            for idx, (id_, titulo, relevancia) in enumerate(fts_resultados, 1):
                print(f"      {idx}. {titulo[:60]} (relevancia: {relevancia:.4f})")
        else:
            print("      No se encontraron coincidencias de texto")
        
        # 2. Búsqueda vectorial (semántica)
        sql_vector = """
            SELECT id, titulo, 1 - (embedding <=> CAST(%s AS vector)) AS similitud
            FROM articulos
            ORDER BY embedding <=> CAST(%s AS vector)
            LIMIT 3;
        """
        print("\n   🧠 RESULTADOS POR SIMILITUD VECTORIAL:")
        cur.execute(sql_vector, (q_vec.tolist(), q_vec.tolist()))
        vec_resultados = cur.fetchall()
        for idx, (id_, titulo, similitud) in enumerate(vec_resultados, 1):
            print(f"      {idx}. {titulo[:60]} (similitud: {similitud:.4f})")
        
        # 3. Búsqueda híbrida (texto + vector)
        sql_hibrido = """
            SELECT 
                id, 
                titulo, 
                ts_rank(ts_doc, to_tsquery('spanish', %s)) AS relevancia_texto,
                1 - (embedding <=> CAST(%s AS vector)) AS similitud_semantica
            FROM articulos
            WHERE ts_doc @@ to_tsquery('spanish', %s)
            ORDER BY (ts_rank(ts_doc, to_tsquery('spanish', %s)) * 0.3 + 
                     (1 - (embedding <=> CAST(%s AS vector))) * 0.7) DESC
            LIMIT 5;
        """
        print("\n   🔀 RESULTADOS HÍBRIDOS (texto 30% + vector 70%):")
        cur.execute(sql_hibrido, (ts_query, q_vec.tolist(), ts_query, ts_query, q_vec.tolist()))
        hibrido_resultados = cur.fetchall()
        if hibrido_resultados:
            for idx, (id_, titulo, relevancia, similitud) in enumerate(hibrido_resultados, 1):
                print(f"      {idx}. {titulo[:60]}")
                print(f"         Texto: {relevancia:.4f} | Vector: {similitud:.4f}")
        else:
            print("      No se encontraron coincidencias")
    
    # Limpiar
    cur.close()
    conn.close()
    print("\n✅ EJERCICIO 4 COMPLETADO")

if __name__ == "__main__":
    main()