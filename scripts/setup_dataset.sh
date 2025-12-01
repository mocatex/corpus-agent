# This script sets up the dataset
# Prerequisites: The dataset of the article-corpus should be downloaded in .parquet format

# Prepare: put your parquet files in the dataset folder -> adapt the `SUBSET_DIR`Variable in the file: `import_subset_to_postgres.py`

echo -e "Setting up the dataset..."

# changing the working directory
cd ../dataset || { echo -e "Directory not found"; exit 1; }

# setup Docker environment
echo -e "Setting up Docker environment..."
docker-compose up -d || { echo -e "Failed to start Docker containers"; exit 1; }
echo -e "Docker environment set up.\n"

# activate the virtual environment
echo -e "Activating virtual environment..."
source ../venv/bin/activate || { echo -e "Failed to activate virtual environment"; exit 1; }
echo -e "Virtual environment activated.\n"

# setup DB schema in Docker
echo -e "Setting up DB schema in Docker..."
docker exec -i corpus_postgres psql -U corpus -d corpus_db < schema.sql || { echo -e "Failed to set up DB schema"; exit 1; }
echo -e "DB schema set up successfully.\n"

# import the subset into PostgreSQL
echo -e "Importing subset into PostgreSQL..."
python3 import_subset_to_postgres.py || { echo -e "Failed to import subset into PostgreSQL"; exit 1; }
echo -e "Subset imported successfully.\n"

echo -e "PostGres database is set up with the dataset."

echo -e "Setting up OpenSearch index..."

# setup OpenSearch table
echo -e "Creating OpenSearch Table..."
curl -k -u admin:'VerySecurePassword123!' \
  -X PUT "https://localhost:9200/article-corpus-opensearch" \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "analysis": {
        "analyzer": {
          "default": {
            "type": "standard"
          }
        }
      }
    },
    "mappings": {
      "properties": {
        "id":           { "type": "integer" },
        "year":         { "type": "integer" },
        "source_domain":{ "type": "keyword" },
        "title":        { "type": "text" },
        "body":         { "type": "text" }
      }
    }
  }' || { echo -e "Failed to create OpenSearch table"; exit 1; }

echo -e "OpenSearch table created successfully.\n"

# import data into OpenSearch
echo -e "Importing data into OpenSearch..."
python3 setup_opensearch.py || { echo -e "Failed to import data"; exit 1; }
echo -e "Data imported into OpenSearch successfully.\n"

echo -e "OpenSearch index is set up with the dataset."
echo -e "Dataset setup completed successfully. You can start using CorpusAgent!"