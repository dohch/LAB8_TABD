

```markdown
# 🧠 Laboratorio 08 - Búsqueda Vectorial con PostgreSQL + pgvector

## 📋 Información General

| Campo | Valor |
|-------|-------|
| **Asignatura** | Tópicos Avanzados de Bases de Datos |
| **Título** | Práctica N° 8: Búsqueda Vectorial con PostgreSQL + pgvector — SQL Semántico y Búsqueda Híbrida |
| **Número de Práctica** | 08 |
| **Año Lectivo** | 2026A |
| **Semestre** | IX |
| **Fecha de Presentación** | 06/07/2026 |
| **Docente** | Mg. Antonio Arroyo Paz |

---

## 👨‍🎓 Integrantes

| Apellidos y Nombres | Código |
|---------------------|--------|
| Huamani Chañi Diego Oswaldo | ... |

---

## 🎯 Objetivos del Laboratorio

1. Instalar y configurar PostgreSQL con la extensión pgvector usando Docker.
2. Diseñar un esquema relacional que integre datos tradicionales con columnas VECTOR.
3. Implementar búsqueda de vecinos más cercanos usando operadores de pgvector.
4. Construir índices IVFFlat y HNSW y comparar rendimiento y recall.
5. Implementar búsqueda híbrida combinando similitud vectorial con filtros SQL.
6. Comparar pgvector con ChromaDB y FAISS.

---

## 🏗️ Estructura del Proyecto

```
lab08-pgvector-renace/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── sql/
│   └── schema.sql
│
├── scripts/
│   ├── insertar_datos.py
│   ├── ejercicio2_indices.py
│   ├── ejercicio3_funcion_sql.py
│   ├── ejercicio4_hibrido.py
│   ├── prueba_busqueda.py
│   └── test_conexion.py
│
├── docs/
│   └── LAB_TABD_8.pdf
│
└── resultados/
    └── capturas/
```

---

## 🐳 Configuración del Entorno

### Requisitos Previos

| Herramienta | Versión |
|-------------|---------|
| Docker | >= 24.0 |
| Python | 3.10+ |
| PostgreSQL | 16 (con pgvector) |

### Instalación de Dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install psycopg2-binary pgvector sentence-transformers numpy pandas matplotlib
```

### Levantar PostgreSQL con pgvector

```bash
docker run -d \
  --name pgvector-lab \
  -e POSTGRES_USER=bdv \
  -e POSTGRES_PASSWORD=bdv2026 \
  -e POSTGRES_DB=bdvdb \
  -p 5433:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

docker ps
docker exec -it pgvector-lab psql -U bdv -d bdvdb -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

---

## 📝 Ejercicios Desarrollados

### ✅ Ejercicio 1: Corpus completo de 30+ documentos RENACE

**Objetivo:** Insertar 30 documentos (10 por categoría: EDA, Dengue, Tuberculosis) con sus embeddings.

**Código clave:**

```python
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    host='localhost',
    port='5433',
    dbname='bdvdb',
    user='bdv',
    password='bdv2026'
)
register_vector(conn)

model = SentenceTransformer('all-MiniLM-L6-v2')
textos = [f"{d['titulo']}. {d['resumen']}" for d in corpus_renace]
embs = model.encode(textos, normalize_embeddings=True).astype(np.float32)

insert_sql = """
INSERT INTO articulos (titulo, resumen, categoria, año, fuente, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
"""
for i, doc in enumerate(corpus_renace):
    cur.execute(insert_sql, (doc['titulo'], doc['resumen'], doc['categoria'],
                             doc['año'], doc['fuente'], embs[i]))
conn.commit()
```

**Resultados:**

- Total de registros: 30
- Dimensión de cada vector: 384
- Documentos: 10 EDA, 10 Dengue, 10 Tuberculosis

---

### ✅ Ejercicio 2: Índice HNSW y comparación con IVFFlat

**Objetivo:** Crear índices IVFFlat y HNSW y comparar rendimiento y recall.

**Índices creados:**

```sql
CREATE INDEX idx_articulos_embedding_ivfflat 
ON articulos 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 6);

CREATE INDEX idx_articulos_embedding_hnsw 
ON articulos 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);
```

**Resultados:**

| Métrica | Exacto | IVFFlat | HNSW |
|---------|--------|---------|------|
| Tiempo (ms) | 5.54 | 4.72 | 5.70 |
| Speedup | 1.0x | 1.2x | 1.0x |
| Recall@5 | 1.00 | 1.00 | 1.00 |

**Análisis:**

- Con 30 documentos, el speedup es pequeño (1.2x) por el overhead del índice.
- Ambos índices mantienen recall perfecto (1.0) en corpus pequeño.
- Con 1M documentos, el speedup sería ~100x.

---

### ✅ Ejercicio 3: Función SQL de búsqueda semántica

**Objetivo:** Crear una función PostgreSQL que encapsule la búsqueda semántica.

**Función creada:**

```sql
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
```

**Llamada desde Python:**

```python
sql = """
    SELECT * FROM buscar_semantico(
        CAST(%s AS vector),
        %s,
        %s
    );
"""
cur.execute(sql, (q_vec.tolist(), 5, filtro))
resultados = cur.fetchall()
```

**Pruebas realizadas:**

| Query | Filtro | Top-1 | Similitud |
|-------|--------|-------|-----------|
| tratamiento de tuberculosis con nuevos medicamentos | Ninguno | TB en poblaciones indígenas | 0.6151 |
| factores de riesgo en enfermedades diarreicas | EDA | EDA en menores de 5 años | 0.5974 |
| control del mosquito Aedes aegypti | Dengue | Nuevas técnicas de control biológico | 0.8368 |
| resistencia a antibióticos | Ninguno | Resistencia a antibióticos en bacterias | 0.7422 |

---

### ✅ Ejercicio 4: Full-Text Search + Búsqueda Vectorial

**Objetivo:** Combinar búsqueda de texto completo con búsqueda vectorial.

**Estructura:**

```sql
ALTER TABLE articulos ADD COLUMN ts_doc TSVECTOR
GENERATED ALWAYS AS (to_tsvector('spanish', titulo || ' ' || resumen)) STORED;

CREATE INDEX idx_articulos_ts ON articulos USING gin(ts_doc);
```

**Consulta híbrida:**

```sql
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
```

**Resultados:**

| Query | Texto completo | Vectorial | Híbrido |
|-------|---------------|-----------|---------|
| rotavirus en niños | Encontró 1 (0.0165) | Encontró 3, top: 0.6629 | Encontró 1 |
| tratamiento de tuberculosis | No encontró nada | Encontró 3, top: 0.6300 | No encontró nada |
| control de mosquitos | Encontró 1 (0.0792) | Encontró 3, top: 0.6511 | Encontró 1 |

**Conclusiones:**

1. El texto completo es literal: falla con variaciones.
2. La búsqueda vectorial entiende el contexto: funciona con sinónimos.
3. La búsqueda híbrida funciona mejor cuando ambos métodos encuentran algo.

---

### ✅ Ejercicio 5: Análisis comparativo y recomendación

**1. pgvector vs ChromaDB:**

- **pgvector** cuando los datos ya están en PostgreSQL y se necesitan JOINs complejos, transacciones ACID y seguridad RLS.
- **ChromaDB** para prototipos rápidos sin relaciones complejas.

**2. FAISS vs pgvector:**

- **FAISS** para ultra-escala (>100M vectores) con GPU y latencia <5ms.
- **pgvector** para sistemas con filtros SQL, ACID, JOINs y datos ya en PostgreSQL.

**3. Impacto del IVFFlat con 30 documentos:**

- Speedup: 1.2x (no justifica el índice en corpus pequeño).
- Con 1M documentos: 100x (justifica ampliamente el índice).

**4. Ventaja de búsqueda híbrida de pgvector:**

- Filtros SQL aplicados ANTES de la búsqueda vectorial (espacio reducido).
- FAISS filtra DESPUÉS (busca en todos los documentos).

**5. Recomendación para RENACE con 5M reportes:**

- **pgvector** por: integración con PostgreSQL existente, ACID, seguridad RLS, filtros SQL, curva de aprendizaje baja.

---

## 📊 Tabla Comparativa: pgvector vs ChromaDB vs FAISS

| Característica | pgvector (PostgreSQL) | ChromaDB | FAISS |
|----------------|----------------------|----------|-------|
| **Tipo** | RDBMS + vectorial | BDV dedicada | Librería de índices |
| **Persistencia** | Automática (ACID) | Ephemeral/Persistent | Manual |
| **SQL/JOINs** | Sí, completo | No | No |
| **Transacciones ACID** | Sí | No | No |
| **Escalabilidad** | ~100M con sharding | ~10M documentos | Miles de millones (GPU) |
| **Latencia (1M)** | 5-50ms | 2-20ms | 0.1-5ms |
| **Índices** | IVFFlat, HNSW | HNSW | Flat, IVF, PQ, HNSW, GPU |
| **Instalación** | Docker o servidor | pip install | pip install |
| **Costo** | Open source | Open source | Open source |
| **Seguridad RLS** | Sí | No | No |
| **Auditoría** | Sí | No | No |

---

## 🔍 Cuestionario

### 1. Diferencia entre operadores `<->`, `<=>`, y `<#>`?

| Operador | Métrica | Uso | Orden |
|----------|---------|-----|-------|
| `<->` | Distancia Euclidiana (L2) | Imágenes, features CNN | ASC |
| `<=>` | Distancia Coseno | Texto, NLP, embeddings normalizados | ASC |
| `<#>` | Producto Interno negado | Vectores normalizados, máxima similitud | ASC |

### 2. ¿Por qué es necesario `register_vector(conn)`?

Registra el adaptador que convierte arrays numpy al tipo VECTOR de PostgreSQL.

**Error sin register_vector():**

```
TypeError: can't adapt type 'numpy.ndarray'
```

### 3. Diferencia entre IVFFlat y HNSW

| Característica | IVFFlat | HNSW |
|----------------|---------|------|
| Requiere datos previos | Sí | No |
| Velocidad construcción | Más rápido | Más lento |
| Memoria | Menos | Más |
| Recall típico | 0.85-0.99 | 0.95-0.99 |

### 4. ¿Qué hace `ivfflat.probes`?

Controla el número de celdas Voronoi a explorar.

- **Probes alto** → mayor recall → mayor latencia
- **Probes bajo** → menor recall → menor latencia

### 5. `lists=100`, `probes=1`: ¿qué porcentaje se explora?

- 1/100 = 1% del espacio de vectores
- Recall esperado: ~70-80%

### 6. ¿Qué es ACID y por qué es relevante?

- **Atomicity**: Todo o nada
- **Consistency**: Datos siempre válidos
- **Isolation**: Transacciones concurrentes aisladas
- **Durability**: Datos persisten ante fallos

**Relevancia en registros médicos:** Integridad de expedientes, consistencia de datos, aislamiento para múltiples médicos, durabilidad ante fallos.

### 7. Memoria: pgvector vs FAISS (100,000 vectores x 384 dims)

- **pgvector**: ~153.6 MB + overhead PostgreSQL (~200-250 MB total)
- **FAISS**: ~153.6 MB (sin overhead adicional)

**Conclusión:** FAISS usa menos RAM.

### 8. Ventaja de `GENERATED ALWAYS AS STORED`

- Calcula tsvector UNA VEZ al insertar/actualizar.
- Búsquedas más rápidas (ya precomputado).
- Permite índice GIN sobre el campo.

### 9. ¿Por qué `<#>` retorna producto interno negado?

Para que `ORDER BY ASC` devuelva primero los vectores más similares.

**Conversión a similitud positiva:**

```sql
SELECT -(embedding <#> query_vec) AS similitud
SELECT 1 - (embedding <=> query_vec) AS similitud
```

### 10. Esquema para expedientes clínicos

```sql
CREATE TABLE expedientes_clinicos (
    id SERIAL PRIMARY KEY,
    paciente_id INTEGER NOT NULL,
    diagnostico TEXT NOT NULL,
    fecha_consulta DATE NOT NULL,
    medico_responsable VARCHAR(100) NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_paciente FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);

CREATE INDEX idx_expedientes_paciente ON expedientes_clinicos(paciente_id);
CREATE INDEX idx_expedientes_fecha ON expedientes_clinicos(fecha_consulta);
CREATE INDEX idx_expedientes_diagnostico ON expedientes_clinicos USING GIN (to_tsvector('spanish', diagnostico));
CREATE INDEX idx_expedientes_embedding ON expedientes_clinicos USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## 🚀 Cómo Ejecutar

```bash
git clone https://github.com/tu-usuario/lab08-pgvector-renace.git
cd lab08-pgvector-renace

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

docker run -d --name pgvector-lab -e POSTGRES_USER=bdv -e POSTGRES_PASSWORD=bdv2026 -e POSTGRES_DB=bdvdb -p 5433:5432 -v pgvector_data:/var/lib/postgresql/data pgvector/pgvector:pg16

docker exec -it pgvector-lab psql -U bdv -d bdvdb -f sql/schema.sql

python scripts/insertar_datos.py
python scripts/ejercicio2_indices.py
python scripts/ejercicio3_funcion_sql.py
python scripts/ejercicio4_hibrido.py
```

---

## 📚 Referencias

1. [pgvector Documentation](https://github.com/pgvector/pgvector)
2. [PostgreSQL Documentation 16](https://www.postgresql.org/docs/16/)
3. [Docker Documentation](https://docs.docker.com/)
4. [Sentence Transformers](https://www.sbert.net/)
5. Douze, M. et al. (2024). The FAISS Library. arXiv:2401.08281
6. Malkov, Y. & Yashunin, D. (2018). HNSW. IEEE TPAMI

---

## ✍️ Autor

**Huamani Chañi Diego Oswaldo**  
Estudiante de Ingeniería de Sistemas - UNSA

---

## 🏆 Conclusiones

pgvector es la solución más completa para sistemas que requieren combinar datos transaccionales con búsqueda semántica, ya que integra operadores vectoriales dentro de PostgreSQL permitiendo consultas híbridas (filtros SQL + similitud de vectores) en una sola operación atómica con ACID, superando a ChromaDB en integridad de datos y a FAISS en facilidad de uso, aunque su rendimiento bruto es inferior a FAISS, lo que lo hace ideal para sistemas como RENACE donde la consistencia de datos es prioritaria sobre la velocidad extrema.
```

---

**¡LISTO!** Eso es todo. Copia ese bloque completo, pégalo en tu `README.md` y súbelo a GitHub. Todo está en formato Markdown, con tablas, código, emojis y estructura profesional.
