# Dataset API Endpoints

Base URL: `/api/v1/datasets`

## Endpoints

### 1. Upload Dataset
```
POST /api/v1/datasets/upload
```

**Description:** Upload a new dataset file

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload (CSV, XLSX, XLS, JSON)
- Max size: 100MB

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "name": "dataset_name",
  "file_path": "/path/to/file.csv",
  "shape": {
    "rows": 1000,
    "cols": 10
  },
  "dtypes": {
    "column1": "int64",
    "column2": "object"
  },
  "missing_values": {
    "column1": 0,
    "column2": 5
  },
  "uploaded_at": "2025-12-24T23:00:00"
}
```

---

### 2. List Datasets
```
GET /api/v1/datasets/
```

**Description:** Get all datasets for the current user

**Query Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100)

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "user_id": "uuid",
    "name": "dataset_name",
    "file_path": "/path/to/file.csv",
    "shape": {"rows": 1000, "cols": 10},
    "dtypes": {...},
    "missing_values": {...},
    "uploaded_at": "2025-12-24T23:00:00"
  }
]
```

---

### 3. Get Dataset by ID
```
GET /api/v1/datasets/{dataset_id}
```

**Description:** Get a specific dataset by ID

**Path Parameters:**
- `dataset_id`: UUID of the dataset

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "name": "dataset_name",
  "file_path": "/path/to/file.csv",
  "shape": {"rows": 1000, "cols": 10},
  "dtypes": {...},
  "missing_values": {...},
  "uploaded_at": "2025-12-24T23:00:00"
}
```

**Errors:**
- `404 Not Found`: Dataset not found or doesn't belong to user

---

### 4. Get Dataset Preview
```
GET /api/v1/datasets/{dataset_id}/preview
```

**Description:** Get a preview of the dataset with sample rows

**Path Parameters:**
- `dataset_id`: UUID of the dataset

**Query Parameters:**
- `rows` (optional): Number of rows to return (default: 10)

**Response:** `200 OK`
```json
{
  "preview": [
    [25, "John", true],
    [30, "Jane", false],
    [35, "Bob", true]
  ],
  "columns": [
    {
      "name": "age",
      "dataType": "int64",
      "nullCount": 0,
      "uniqueCount": 50,
      "sampleValues": [25, 30, 35, 40, 45]
    },
    {
      "name": "name",
      "dataType": "object",
      "nullCount": 2,
      "uniqueCount": 98,
      "sampleValues": ["John", "Jane", "Bob", "Alice", "Charlie"]
    },
    {
      "name": "active",
      "dataType": "bool",
      "nullCount": 0,
      "uniqueCount": 2,
      "sampleValues": [true, false]
    }
  ],
  "totalRows": 1000,
  "displayedRows": 10
}
```

**Errors:**
- `404 Not Found`: Dataset not found or file doesn't exist on disk
- `500 Internal Server Error`: Failed to read dataset

---

### 5. Get Dataset Statistics
```
GET /api/v1/datasets/{dataset_id}/stats
```

**Description:** Get comprehensive statistics for the dataset

**Path Parameters:**
- `dataset_id`: UUID of the dataset

**Response:** `200 OK`
```json
{
  "rowCount": 1000,
  "columnCount": 10,
  "numericColumns": 5,
  "categoricalColumns": 3,
  "missingValues": 15,
  "duplicateRows": 2,
  "memoryUsage": 80000,
  "columns": [
    {
      "name": "age",
      "dataType": "int64",
      "nullCount": 0,
      "uniqueCount": 50,
      "sampleValues": [25, 30, 35, 40, 45]
    },
    {
      "name": "name",
      "dataType": "object",
      "nullCount": 2,
      "uniqueCount": 98,
      "sampleValues": ["John", "Jane", "Bob", "Alice", "Charlie"]
    }
  ]
}
```

**Errors:**
- `404 Not Found`: Dataset not found or file doesn't exist on disk
- `500 Internal Server Error`: Failed to read dataset

---

### 6. Delete Dataset
```
DELETE /api/v1/datasets/{dataset_id}
```

**Description:** Delete a dataset

**Path Parameters:**
- `dataset_id`: UUID of the dataset

**Response:** `204 No Content`

**Errors:**
- `404 Not Found`: Dataset not found or doesn't belong to user

---

## Data Types

### ColumnInfo
```typescript
{
  name: string;
  dataType: string;  // e.g., "int64", "float64", "object", "bool"
  nullCount: number;
  uniqueCount: number;
  sampleValues: any[];  // First 5 non-null values
}
```

### DatasetShape
```typescript
{
  rows: number;
  cols: number;
}
```

---

## Common Error Responses

### 400 Bad Request
```json
{
  "detail": "File type not supported. Allowed: .csv, .xlsx, .xls, .json"
}
```

### 404 Not Found
```json
{
  "detail": "Dataset {dataset_id} not found"
}
```

### 413 Request Entity Too Large
```json
{
  "detail": "File size exceeds maximum allowed size of 100MB"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to read dataset: {error_message}"
}
```

---

## Authentication

Currently using mock authentication that returns a hardcoded user ID:
```python
user_id = "00000000-0000-0000-0000-000000000001"
```

**TODO:** Replace with actual JWT authentication

---

## File Storage

- Files are stored in: `uploads/{user_id}/{dataset_id}/{filename}`
- Supported formats: CSV, XLSX, XLS, JSON
- Maximum file size: 100MB
- Files are automatically deleted when the dataset is deleted

---

## Example Usage

### Upload a Dataset
```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@dataset.csv"
```

### Get Dataset Preview
```bash
curl http://localhost:8000/api/v1/datasets/{dataset_id}/preview?rows=20
```

### Get Dataset Statistics
```bash
curl http://localhost:8000/api/v1/datasets/{dataset_id}/stats
```

### Delete Dataset
```bash
curl -X DELETE http://localhost:8000/api/v1/datasets/{dataset_id}
```

---

*Last Updated: 2025-12-24*
