"""A miniature corpus for hermetic end-to-end tests.

Synthetic, but shaped like the real thing: structured abstracts with section labels,
exact identifiers (drug names, doses, thresholds, trial ids) that reward BM25, and
paraphrasable mechanism statements that reward dense retrieval. Small enough that CI
can build an index and run the full eval in seconds, with no network access.
"""

from __future__ import annotations

from ragmed.types import Document, Section


def _doc(pmid: str, title: str, sections: list[tuple[str, str]], year: str, mesh: list[str]) -> Document:
    return Document(
        doc_id=f"pmid:{pmid}",
        title=title,
        source_type="pubmed",
        sections=[Section(heading=h, text=t) for h, t in sections],
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        date=year,
        meta={"pmid": pmid, "journal": "Journal of Test Medicine", "mesh_terms": mesh},
    )


def mini_corpus() -> list[Document]:
    return [
        _doc(
            "10000001",
            "Empagliflozin in Type 2 Diabetes: Glycaemic and Renal Outcomes",
            [
                ("Background", "Sodium glucose cotransporter 2 inhibitors reduce renal glucose reabsorption in the proximal tubule, producing glycosuria independent of insulin secretion."),
                ("Methods", "In trial NCT01131676 we randomised 4687 adults with type 2 diabetes to empagliflozin 10 mg once daily or placebo for 52 weeks."),
                ("Results", "Empagliflozin 10 mg reduced HbA1c by 0.62 percentage points compared with placebo. Estimated glomerular filtration rate declined more slowly in the treatment arm."),
                ("Conclusions", "Empagliflozin 10 mg once daily improves glycaemic control and slows renal function decline in type 2 diabetes."),
            ],
            "2022",
            ["Diabetes Mellitus, Type 2", "Sodium-Glucose Transporter 2 Inhibitors", "Glycated Hemoglobin"],
        ),
        _doc(
            "10000002",
            "Metformin as First Line Therapy in Newly Diagnosed Type 2 Diabetes",
            [
                ("Background", "Metformin decreases hepatic gluconeogenesis and improves peripheral insulin sensitivity without causing hypoglycaemia as monotherapy."),
                ("Methods", "We reviewed 18 randomised trials of metformin monotherapy in adults with newly diagnosed type 2 diabetes."),
                ("Results", "Metformin lowered HbA1c by a mean of 1.12 percentage points. Gastrointestinal intolerance occurred in 22% of participants and led to discontinuation in 4%."),
                ("Conclusions", "Metformin remains the recommended initial pharmacologic treatment for type 2 diabetes in the absence of contraindications."),
            ],
            "2021",
            ["Diabetes Mellitus, Type 2", "Metformin", "Hypoglycemic Agents"],
        ),
        _doc(
            "10000003",
            "Diagnostic Thresholds for Diabetes and Prediabetes",
            [
                ("Abstract", "Diabetes is diagnosed at a fasting plasma glucose of 126 mg/dL or higher, a 2-hour value of 200 mg/dL or higher during an oral glucose tolerance test, or an HbA1c of 6.5% or higher. Prediabetes corresponds to an HbA1c between 5.7% and 6.4%. Coded as ICD-10 E11 for type 2 disease."),
            ],
            "2023",
            ["Diabetes Mellitus, Type 2", "Blood Glucose", "Glycated Hemoglobin"],
        ),
        _doc(
            "10000004",
            "Sacubitril/Valsartan in Heart Failure with Preserved Ejection Fraction",
            [
                ("Background", "Angiotensin receptor neprilysin inhibition increases natriuretic peptide availability by blocking their enzymatic degradation."),
                ("Methods", "We enrolled 4822 patients with an ejection fraction of at least 45% and randomised them to sacubitril/valsartan 97/103 mg twice daily or valsartan alone."),
                ("Results", "The primary composite endpoint occurred in 894 patients receiving sacubitril/valsartan versus 1009 receiving valsartan, a rate ratio of 0.87."),
                ("Conclusions", "Sacubitril/valsartan did not significantly reduce total hospitalisations and cardiovascular death in HFpEF."),
            ],
            "2020",
            ["Heart Failure", "Stroke Volume", "Neprilysin"],
        ),
        _doc(
            "10000005",
            "Diuretic Strategy in Acute Decompensated Heart Failure",
            [
                ("Background", "Loop diuretics inhibit the sodium potassium chloride cotransporter in the thick ascending limb, producing rapid natriuresis."),
                ("Methods", "Patients admitted with acute decompensated heart failure received furosemide by bolus every 12 hours or by continuous infusion."),
                ("Results", "There was no significant difference in symptom relief at 72 hours. Serum creatinine rose by more than 0.3 mg/dL in 14% of the bolus group."),
                ("Conclusions", "Bolus and continuous furosemide produce comparable decongestion in acute heart failure."),
            ],
            "2019",
            ["Heart Failure", "Furosemide", "Diuretics"],
        ),
        _doc(
            "10000006",
            "Amoxicillin versus Amoxicillin-Clavulanate for Community Acquired Pneumonia",
            [
                ("Background", "Streptococcus pneumoniae remains the most common bacterial cause of community acquired pneumonia in ambulatory adults."),
                ("Methods", "Adults with radiographically confirmed community acquired pneumonia received amoxicillin 1 g three times daily or amoxicillin-clavulanate for 5 days."),
                ("Results", "Clinical cure at day 15 was 89.7% with amoxicillin and 91.2% with amoxicillin-clavulanate. Diarrhoea was more frequent with clavulanate."),
                ("Conclusions", "Amoxicillin 1 g three times daily for 5 days is adequate for uncomplicated community acquired pneumonia in adults."),
            ],
            "2022",
            ["Pneumonia", "Amoxicillin", "Anti-Bacterial Agents"],
        ),
        _doc(
            "10000007",
            "CURB-65 Score and Site of Care Decisions in Pneumonia",
            [
                ("Abstract", "The CURB-65 score assigns one point each for confusion, urea above 7 mmol/L, respiratory rate of 30 or more, blood pressure below 90 systolic, and age 65 or older. A score of 0 to 1 supports outpatient management, 2 suggests admission, and 3 or more indicates severe pneumonia requiring consideration of intensive care."),
            ],
            "2021",
            ["Pneumonia", "Severity of Illness Index"],
        ),
        _doc(
            "10000008",
            "Apixaban for Stroke Prevention in Atrial Fibrillation",
            [
                ("Background", "Direct factor Xa inhibition prevents thrombin generation without requiring routine coagulation monitoring."),
                ("Methods", "Patients with nonvalvular atrial fibrillation received apixaban 5 mg twice daily or warfarin titrated to an INR of 2.0 to 3.0."),
                ("Results", "Stroke or systemic embolism occurred at 1.27% per year with apixaban versus 1.60% per year with warfarin. Major bleeding was 2.13% versus 3.09% per year."),
                ("Conclusions", "Apixaban 5 mg twice daily was superior to warfarin for stroke prevention with less major bleeding."),
            ],
            "2018",
            ["Atrial Fibrillation", "Anticoagulants", "Stroke"],
        ),
        _doc(
            "10000009",
            "CHA2DS2-VASc Scoring for Anticoagulation Decisions",
            [
                ("Abstract", "The CHA2DS2-VASc score assigns two points each for age 75 or older and prior stroke, and one point each for congestive heart failure, hypertension, age 65 to 74, diabetes, vascular disease, and female sex. Oral anticoagulation is recommended at a score of 2 or more in men and 3 or more in women."),
            ],
            "2020",
            ["Atrial Fibrillation", "Anticoagulants", "Risk Assessment"],
        ),
        _doc(
            "10000010",
            "ACE Inhibition and Progression of Chronic Kidney Disease",
            [
                ("Background", "Angiotensin converting enzyme inhibitors reduce intraglomerular pressure by preferentially dilating the efferent arteriole."),
                ("Methods", "Adults with chronic kidney disease and proteinuria above 1 g per day received ramipril 10 mg daily or placebo."),
                ("Results", "The rate of decline in glomerular filtration rate was 0.53 mL/min per month with ramipril versus 0.88 with placebo. Proteinuria fell by 55%."),
                ("Conclusions", "Ramipril 10 mg daily slows progression of proteinuric chronic kidney disease."),
            ],
            "2019",
            ["Renal Insufficiency, Chronic", "Angiotensin-Converting Enzyme Inhibitors", "Proteinuria"],
        ),
        _doc(
            "10000011",
            "Staging of Chronic Kidney Disease by Estimated GFR",
            [
                ("Abstract", "Chronic kidney disease stage G3a corresponds to an estimated glomerular filtration rate of 45 to 59 mL/min/1.73 m2, stage G3b to 30 to 44, stage G4 to 15 to 29, and stage G5 to below 15. Albuminuria categories A1 through A3 correspond to urine albumin to creatinine ratios below 30, 30 to 300, and above 300 mg/g."),
            ],
            "2022",
            ["Renal Insufficiency, Chronic", "Glomerular Filtration Rate", "Albuminuria"],
        ),
        _doc(
            "10000012",
            "Dapagliflozin in Chronic Kidney Disease Without Diabetes",
            [
                ("Background", "SGLT2 inhibition reduces intraglomerular pressure through tubuloglomerular feedback, an effect independent of glycaemic control."),
                ("Methods", "Participants with chronic kidney disease, with and without type 2 diabetes, received dapagliflozin 10 mg daily or placebo."),
                ("Results", "The primary composite kidney endpoint occurred in 9.2% of the dapagliflozin group and 14.5% of the placebo group. Benefit was consistent in participants without diabetes."),
                ("Conclusions", "Dapagliflozin 10 mg daily slows chronic kidney disease progression regardless of diabetes status."),
            ],
            "2021",
            ["Renal Insufficiency, Chronic", "Sodium-Glucose Transporter 2 Inhibitors", "Diabetes Mellitus, Type 2"],
        ),
    ]
