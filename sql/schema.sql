-- =====================================================
-- LABORATORIO 8: pgvector - Búsqueda Vectorial
-- Esquema completo de la base de datos
-- =====================================================

-- 1. Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Crear tabla principal de artículos
CREATE TABLE articulos (
    id SERIAL PRIMARY KEY,
    titulo TEXT NOT NULL,
    resumen TEXT NOT NULL,
    categoria VARCHAR(50) NOT NULL,
    año INTEGER,
    fuente VARCHAR(100),
    embedding VECTOR(384)  -- all-MiniLM-L6-v2 → 384 dimensiones
);

-- 3. Índices para búsquedas híbridas
CREATE INDEX idx_articulos_categoria ON articulos(categoria);
CREATE INDEX idx_articulos_año ON articulos(año);

-- 4. Índice IVFFlat (búsqueda aproximada)
CREATE INDEX idx_articulos_embedding_ivfflat 
ON articulos 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 6);  -- sqrt(30) ≈ 6

-- 5. Índice HNSW (Hierarchical Navigable Small World)
CREATE INDEX idx_articulos_embedding_hnsw 
ON articulos 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);

-- 6. Columna para Full-Text Search
ALTER TABLE articulos ADD COLUMN ts_doc TSVECTOR
GENERATED ALWAYS AS (to_tsvector('spanish', titulo || ' ' || resumen)) STORED;

-- 7. Índice GIN para búsqueda de texto completo
CREATE INDEX idx_articulos_ts ON articulos USING gin(ts_doc);

-- 8. Función de búsqueda semántica
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

-- 9. Verificación de la función
\df buscar_semantico

-- 10. Consulta de ejemplo
SELECT * FROM buscar_semantico(
    '[0.1, 0.2, -0.3, ...]'::vector, 
    5, 
    'Tuberculosis'
);
