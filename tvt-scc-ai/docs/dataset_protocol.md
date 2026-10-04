# Dataset Protocol

Guidelines for dataset curation, annotation, and split protocols.

# Dataset Collection Protocol  
## Canine TVT vs SCC Cytology Image Classification Project

### 1. Purpose

This document defines the protocol for collecting, organizing, labeling, and quality-controlling canine cytology images used in the development of an artificial intelligence model for differentiating:

- Canine Transmissible Venereal Tumour (TVT)
- Canine Squamous Cell Carcinoma (SCC)

The purpose of this protocol is to ensure that images used for model development are collected consistently, are diagnostically reliable, and are suitable for machine-learning research.

The protocol is intended to be followed before model training begins.

---

# 2. Study Unit

The primary biological unit of the dataset is the **individual dog/case**, not the individual image.

One dog with multiple cytology images is considered:

**One independent case with multiple images.**

For example:

```text
Case: TVT001

TVT001_001.jpg
TVT001_002.jpg
TVT001_003.jpg
TVT001_004.jpg
TVT001_005.jpg
```

These five images represent **one case**, not five independent cases.

All images from a single dog must remain within the same dataset partition during machine-learning experiments.

Images from the same case must never be divided between training, validation, and testing datasets.

This is necessary to prevent **data leakage**.

---

# 3. Target Classes

The initial dataset will contain two diagnostic classes:

```text
TVT
SCC
```

The project should initially be treated as a **binary classification problem**.

Images from other tumour types, inflammatory lesions, normal tissue, or diagnostically uncertain lesions should not be assigned to either class.

Such images may be stored separately for possible future research.

---

# 4. Case Identification

Every included dog must receive a unique anonymized case identifier.

Recommended format:

```text
TVT001
TVT002
TVT003

SCC001
SCC002
SCC003
```

The identifier should not contain:

- Owner name
- Owner telephone number
- Owner address
- Veterinary registration number that can directly identify the owner
- Any other unnecessary personally identifiable information

The machine-learning dataset should contain only information necessary for the research.

---

# 5. Diagnostic Confirmation

Only cases with a sufficiently reliable diagnosis should be included in the final research dataset.

## 5.1 TVT cases

TVT cases should preferably have a diagnosis confirmed by an experienced veterinarian or veterinary pathologist using cytological examination and the relevant clinical findings.

Where additional confirmation is available, it should be recorded.

Possible confirmation methods may include:

- Cytological diagnosis
- Histopathological confirmation
- Clinical examination combined with characteristic cytology
- Other specialist-confirmed diagnostic procedures

The confirmation method must be recorded in the metadata.

---

## 5.2 SCC cases

SCC cases should preferably have a confirmed diagnosis.

Histopathological confirmation is desirable where available.

Possible confirmation methods include:

- Histopathology
- Cytology confirmed by a veterinary pathologist
- Cytology with supporting clinical and histopathological findings

Because SCC may have cytological appearances that overlap with other lesions, diagnostically uncertain SCC cases should not be included in the primary dataset.

---

# 6. Diagnostic Confidence

Each case should ideally have a diagnostic confidence category.

Recommended categories:

```text
confirmed
probable
uncertain
```

Only:

```text
confirmed
```

cases should be included in the primary training and testing dataset whenever possible.

`probable` and `uncertain` cases should be stored separately.

They may later be useful for testing model behaviour on difficult cases but should not initially be used as ground-truth training labels.

---

# 7. Cytology Only

The primary AI dataset must contain **cytology images**.

Histopathology images must not be mixed into the cytology classification dataset.

For example, the following dataset design is invalid:

```text
TVT → cytology images
SCC → histology images
```

A model trained in this way could learn differences between cytology and histology instead of learning differences between TVT and SCC.

Therefore:

```text
TVT → cytology
SCC → cytology
```

must be maintained.

Histopathology may be used to confirm the diagnosis of a case but should not be used as the image input for the primary classifier.

---

# 8. Cytology Preparation Method

The cytology collection technique should be recorded whenever available.

Examples may include:

- Fine-needle aspiration
- Fine-needle non-aspiration
- Impression smear
- Scraping
- Swab preparation

If multiple collection methods are used, the method should be included in the dataset metadata.

Whenever possible, the same or similar collection method should be used across both diagnostic classes.

This reduces the possibility of the AI model learning collection-method differences rather than tumour morphology.

---

# 9. Staining

The staining method used for each slide must be recorded.

Examples may include:

- Diff-Quik
- Wright-Giemsa
- May-Grünwald-Giemsa
- Other Romanowsky-type stains

Ideally, the same staining protocol should be used for both TVT and SCC cases.

For example:

```text
TVT → Diff-Quik
SCC → Diff-Quik
```

is preferable to:

```text
TVT → Diff-Quik
SCC → Wright-Giemsa
```

If different stains are unavoidable, stain type must be documented.

Staining technique must never be allowed to become a reliable proxy for diagnosis.

---

# 10. Microscope Magnification

Magnification must be recorded for every image.

Recommended magnifications for cytological imaging include:

```text
400×
1000×
```

The final magnification protocol should be agreed with the veterinary/pathology team based on the available microscope and diagnostic practice.

For the initial study, it is preferable to use **one standardized magnification for the main dataset**.

If multiple magnifications are collected, they should be clearly recorded and analyzed separately where necessary.

For example:

```text
TVT001_400x_001.jpg
TVT001_1000x_001.jpg
```

Do not allow one class to contain mostly one magnification and the second class another.

For example:

```text
TVT → mostly 1000×
SCC → mostly 400×
```

could create a serious confounding factor.

---

# 11. Number of Images Per Case

Multiple microscope fields should be captured from every case.

A reasonable initial target is:

```text
5–15 usable images per case
```

A target of approximately:

```text
10 images per case
```

is recommended where possible.

Images should represent different diagnostically useful areas of the smear.

The objective is to capture natural within-case cytological variation rather than repeatedly photographing almost identical fields.

---

# 12. Field Selection

Microscope fields should be selected according to a consistent protocol.

Images should preferably contain:

- Adequate cellularity
- Clearly visible cells
- Representative tumour-cell populations
- Sufficient focus
- Appropriate staining
- Minimal obstruction

Multiple fields should be selected from different regions of the slide where possible.

Researchers should avoid intentionally selecting only the most obvious or "perfect" examples of each tumour.

The dataset should represent realistic diagnostic material.

However, completely non-diagnostic fields should not be included in the primary dataset.

---

# 13. Avoiding Selection Bias

Field selection should not intentionally favour features expected to distinguish TVT and SCC.

For example, researchers should not select TVT images only because they contain obvious cytoplasmic vacuoles.

Similarly, SCC images should not be selected only because they contain obvious keratinization.

The dataset should represent the overall cytological appearance encountered during routine diagnostic examination.

Otherwise, the resulting model may perform well only on artificially selected textbook examples.

---

# 14. Image Acquisition

Images should ideally be captured using the same:

- Microscope
- Camera
- Microscope-camera adapter
- Image acquisition software
- Resolution
- Illumination settings

where practical.

If multiple devices are used, the device information should be recorded.

Variation in imaging hardware is not necessarily undesirable, but it must not be strongly associated with diagnosis.

For example:

```text
Microscope A → all TVT
Microscope B → all SCC
```

must be avoided.

Otherwise, the model may learn equipment-specific differences.

---

# 15. Recommended Image Format

Original images should be retained at their highest practical quality.

Preferred formats:

```text
PNG
TIFF
high-quality JPEG
```

Lossless formats such as PNG or TIFF are preferable where storage capacity permits.

Images should not be repeatedly converted or recompressed.

The original image must always remain unchanged.

---

# 16. Original Images

Original images must be stored under:

```text
data/raw/
```

Example:

```text
data/raw/
├── TVT/
│   ├── TVT001/
│   ├── TVT002/
│   └── TVT003/
│
└── SCC/
    ├── SCC001/
    ├── SCC002/
    └── SCC003/
```

Original images must never be:

- Resized permanently
- Cropped permanently
- Enhanced permanently
- Overwritten
- Colour-adjusted and saved over the original
- Augmented and saved over the original

Any transformed images must be generated separately.

---

# 17. Processed Images

Images generated by preprocessing may be stored under:

```text
data/processed/
```

Examples include:

- Resized images
- Cropped images
- Normalized images
- Quality-controlled images

Processed images must remain traceable to their original image.

---

# 18. Minimum Image Quality

Images included in the main dataset should satisfy the following minimum requirements:

### Focus

Tumour-cell morphology should be sufficiently sharp for visual interpretation.

### Exposure

Images should not be severely overexposed or underexposed.

### Resolution

Resolution must be sufficient to identify relevant cellular structures.

### Staining

Staining should be adequate for cytological interpretation.

### Cell visibility

A diagnostically useful number of cells should be visible.

### Artefacts

Minor artefacts may be accepted if they represent realistic routine samples.

Severe artefacts that make interpretation impossible should be excluded.

---

# 19. Possible Exclusion Criteria

An image may be excluded if it contains:

- Severe blur
- Severe motion artefact
- Extreme overexposure
- Extreme underexposure
- Very poor staining
- No diagnostically useful cells
- Severe contamination
- Major imaging artefacts
- Incorrect magnification information
- Unknown case identity
- Unknown diagnosis
- Corrupted image data
- Duplicate images

The reason for exclusion should preferably be recorded.

---

# 20. Case-Level Exclusion Criteria

An entire case may be excluded if:

- The final diagnosis is uncertain.
- TVT or SCC diagnosis cannot be adequately confirmed.
- Case identification is unreliable.
- Images cannot be reliably associated with the correct animal.
- Image quality is consistently unsuitable.
- Required metadata is missing.
- The case represents another disease or tumour type.
- Ethical or data-use requirements are not satisfied.

Excluded cases should not simply be deleted without documentation.

A record of exclusion and the reason should be maintained.

---

# 21. Image Naming Convention

Each image should have a unique filename.

Recommended format:

```text
{CASE_ID}_{IMAGE_NUMBER}.jpg
```

Examples:

```text
TVT001_001.jpg
TVT001_002.jpg
TVT001_003.jpg

SCC001_001.jpg
SCC001_002.jpg
SCC001_003.jpg
```

If magnification is included:

```text
TVT001_1000x_001.jpg
SCC001_1000x_001.jpg
```

Avoid filenames such as:

```text
image1.jpg
photo_new.jpg
final_image.jpg
IMG_2345.jpg
good_tvt.jpg
```

because these do not reliably connect an image to its metadata.

---

# 22. Required Metadata

The metadata file should be stored at:

```text
metadata/cases.csv
```

Recommended fields include:

```text
case_id
image_id
diagnosis
diagnostic_confidence
confirmation_method
cytology_collection_method
stain
magnification
anatomical_site
image_path
image_quality
included
exclusion_reason
```

Additional information may be added where scientifically useful.

---

# 23. Example Metadata

Example:

```csv
case_id,image_id,diagnosis,diagnostic_confidence,confirmation_method,cytology_collection_method,stain,magnification,anatomical_site,image_path,image_quality,included,exclusion_reason
TVT001,TVT001_001,TVT,confirmed,cytology,FNA,Diff-Quik,1000x,genital,data/raw/TVT/TVT001/TVT001_001.jpg,good,yes,
TVT001,TVT001_002,TVT,confirmed,cytology,FNA,Diff-Quik,1000x,genital,data/raw/TVT/TVT001/TVT001_002.jpg,good,yes,
SCC001,SCC001_001,SCC,confirmed,histopathology,FNA,Diff-Quik,1000x,prepuce,data/raw/SCC/SCC001/SCC001_001.jpg,good,yes,
```

---

# 24. Optional Clinical Metadata

Where available and ethically appropriate, the following may also be collected:

```text
age
sex
breed
anatomical site
lesion location
clinical presentation
date of sample collection
```

These variables should be used only if relevant to the research question.

Unnecessary owner-identifiable data should not be collected.

---

# 25. Data Anonymization

All research images and metadata must be anonymized before being entered into the machine-learning dataset.

The research dataset must not contain:

- Owner names
- Owner addresses
- Telephone numbers
- Personal identification numbers
- Unnecessary clinical identifiers

Only anonymized research case IDs should be used.

If a separate mapping between hospital records and study IDs is required, it should be stored securely and separately from the machine-learning dataset.

---

# 26. Dataset Size Target

The project should aim for approximately:

```text
TVT: 50 independent cases
SCC: 50 independent cases
```

with approximately:

```text
5–15 usable images per case
```

A practical target would therefore be approximately:

```text
100 independent dogs
~1,000 original cytology images
```

Where this target cannot be achieved, a smaller dataset may be used for a pilot or feasibility study.

Approximately:

```text
30 TVT cases
30 SCC cases
```

may be considered a reasonable minimum target for an initial pilot study, subject to statistical justification and the availability of confirmed cases.

The number of independent cases must always be reported separately from the number of images.

---

# 27. Class Balance

An approximately balanced dataset is preferred.

For example:

```text
50 TVT cases
50 SCC cases
```

is preferable to:

```text
100 TVT cases
15 SCC cases
```

If major class imbalance occurs, it should be documented and addressed during model development.

The priority should be obtaining additional independent cases rather than generating artificial duplicates.

---

# 28. Data Augmentation

Image augmentation may later be applied during training.

Possible transformations may include:

- Small rotations
- Horizontal or vertical flipping where biologically appropriate
- Limited cropping
- Limited brightness or contrast variation
- Limited geometric transformations

Augmented images are **not independent biological samples**.

For example:

```text
50 original cases
```

remain:

```text
50 independent cases
```

even if augmentation produces thousands of training images.

Augmented images must never be included in the validation or test sets as substitutes for independent cases.

---

# 29. Dataset Partitioning

Dataset splitting must occur at the **case level**.

A possible split is:

```text
Training:   70%
Validation: 15%
Testing:    15%
```

However, exact splitting may be modified depending on final dataset size.

Example:

```text
TVT001
├── image001
├── image002
├── image003
└── image004
```

If `TVT001` belongs to the training dataset, **all four images must remain in training**.

None may appear in validation or testing.

---

# 30. Data Leakage Prevention

The following must never occur:

```text
TVT001_001 → training
TVT001_002 → training
TVT001_003 → testing
```

because these images originate from the same dog.

This would allow the model to encounter highly related images during both training and testing and may artificially increase measured performance.

Dataset partitioning must therefore use:

```text
case_id
```

rather than individual image filenames.

---

# 31. Duplicate Detection

Exact or near-duplicate images should not be treated as separate independent observations.

Before training, the dataset should be checked for:

- Identical files
- Renamed copies
- Near-identical microscope fields
- Accidentally repeated exports

Duplicate detection procedures should be documented.

---

# 32. Quality-Control Review

Before inclusion in the final dataset, images should ideally undergo review by a qualified veterinarian, veterinary pathologist, or appropriately supervised researcher.

The review should confirm:

1. Correct case identification.
2. Correct diagnostic label.
3. Adequate cytological image quality.
4. Correct magnification metadata where available.
5. Correct stain information where available.
6. Absence of accidental duplication.
7. Suitability for inclusion.

---

# 33. Quality Categories

Images may optionally receive a quality category:

```text
good
acceptable
poor
```

Recommended interpretation:

### Good

Clear focus, adequate cellularity and staining, diagnostically useful.

### Acceptable

Minor imperfections are present, but the image remains interpretable.

### Poor

Insufficient for reliable cytological interpretation.

Images categorized as `poor` should normally be excluded from primary model development.

---

# 34. Dataset Versioning

Each major dataset used in an experiment should be versioned.

Example:

```text
dataset_v1
dataset_v2
dataset_v3
```

A dataset version should not silently change after experiments have been reported.

Changes such as:

- Adding cases
- Removing cases
- Correcting labels
- Changing exclusion criteria

should produce a new version or be recorded in a change log.

---

# 35. Dataset Documentation

For every dataset version, record:

```text
Number of TVT cases
Number of SCC cases
Number of TVT images
Number of SCC images
Stain distribution
Magnification distribution
Image-quality distribution
Excluded cases
Exclusion reasons
Training cases
Validation cases
Testing cases
```

This information should later be used in the research manuscript.

---

# 36. Ethical Considerations

Appropriate institutional, veterinary, and research ethics requirements must be followed before using clinical samples or images.

The research team should confirm:

- Whether institutional ethics approval is required
- Whether retrospective clinical images may be used
- Whether owner consent is required
- Requirements for anonymization
- Requirements for secure storage
- Requirements for publication or data sharing

No data should be publicly released unless permission for that type of sharing has been obtained.

---

# 37. Public Images

Images downloaded from websites, publications, teaching atlases, or other public sources should **not automatically be included in the main clinical dataset**.

Public images may initially be used for:

- Learning
- Testing software
- Debugging image-loading code
- Developing visualization tools
- Demonstrating the prototype

Public images should only enter the research training dataset if:

1. Their diagnosis is reliable.
2. Their reuse license permits the intended use.
3. Their source is documented.
4. Their imaging characteristics are compatible with the research design.
5. Their inclusion does not introduce substantial bias.

The primary evaluation dataset should preferably consist of independently collected and diagnostically confirmed clinical cases.

---

# 38. Synthetic Images

AI-generated synthetic TVT or SCC images should not be treated as independent clinical cases.

Synthetic data should not be used in the primary test dataset.

If synthetic-image experiments are conducted later, they must be reported separately from the primary clinical-image analysis.

---

# 39. Data Storage

Recommended structure:

```text
data/
├── raw/
│   ├── TVT/
│   │   ├── TVT001/
│   │   ├── TVT002/
│   │   └── ...
│   │
│   └── SCC/
│       ├── SCC001/
│       ├── SCC002/
│       └── ...
│
└── processed/
```

Metadata should remain separate:

```text
metadata/
├── cases.csv
└── data_dictionary.md
```

---

# 40. Backup

At least two secure copies of the original dataset should be maintained.

The working copy should not be the only copy of the original images.

Original clinical images should never be permanently altered by preprocessing scripts.

---

# 41. Initial Collection Checklist

Before accepting a case into the dataset, verify:

- [ ] A unique anonymized case ID has been assigned.
- [ ] The dog is an independent case.
- [ ] Diagnosis is TVT or SCC.
- [ ] Diagnosis has been appropriately confirmed.
- [ ] Diagnostic confidence has been recorded.
- [ ] Images are cytology rather than histology.
- [ ] Collection technique is known where available.
- [ ] Staining method has been recorded.
- [ ] Magnification has been recorded.
- [ ] Anatomical site has been recorded where available.
- [ ] Multiple representative fields have been captured.
- [ ] Images are sufficiently focused.
- [ ] Exposure is acceptable.
- [ ] Cell morphology is visible.
- [ ] Files use the correct naming convention.
- [ ] Images are linked to the correct case ID.
- [ ] Required metadata has been recorded.
- [ ] Personally identifiable owner information has been removed.
- [ ] Data-use and ethical requirements have been satisfied.

---

# 42. Primary Dataset Objective

The primary dataset should ultimately represent:

> A collection of anonymized, diagnostically confirmed canine TVT and SCC cytology images collected using documented and reasonably standardized cytological and imaging procedures, with multiple representative images per independent case and strict case-level separation between machine-learning training, validation, and test datasets.

The quality and independence of the cases should be prioritized over simply maximizing the total number of images.