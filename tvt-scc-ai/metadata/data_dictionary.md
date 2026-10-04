# Data Dictionary

| Column Name | Type | Description | Values / Examples |
|---|---|---|---|
| `case_id` | String | Unique identifier for each image/case | `CASE_0001` |
| `patient_id` | String | De-identified animal/patient ID | `P_0102` |
| `diagnosis` | String | Ground truth label | `TVT`, `SCC` |
| `image_filename` | String | Filename or relative path in `data/raw` | `case_0001.jpg` |
| `split` | String | Assigned dataset split | `train`, `val`, `test` |
