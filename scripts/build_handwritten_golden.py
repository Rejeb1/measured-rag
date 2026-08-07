"""Hand-written golden set.

Generated golden sets need a capable writer model. `llama3.2:3b` — the largest that
fits this machine's 4GB card — was not capable enough: across three pilots it yielded
2/8 factoid and 1/4 multi-hop, producing questions like "What is the name of the drug
used in the trial?" that are useless as standalone search queries. The screener
correctly rejected them, which is the eval layer working, but the result was no usable
set.

So this set is written by hand against the real indexed corpus. Every question was
written after reading the chunk it is labelled with, and every gold id is a real
`chunk_id` from `data/index/chunks.jsonl`. That makes it slower to extend but
deterministic, reviewable, and free of any dependency on a local model.

Question types follow the design: factoid (one chunk), multi-hop (two chunks from
different documents), aggregation (several chunks), and unanswerable (no chunk).

The unanswerable questions are deliberately *topically adjacent* — same disease area,
same drug class — so retrieval confidently surfaces passages and the generator has to
decide to refuse. An obviously off-topic question would make abstention trivially
correct and measure nothing (see the BM25 out-of-vocabulary finding in the README).

Run:  python scripts/build_handwritten_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragmed.config import Config  # noqa: E402
from ragmed.index.store import CorpusIndex  # noqa: E402
from ragmed.store import save_golden  # noqa: E402
from ragmed.types import GoldenItem  # noqa: E402

# (qid, question, gold_chunk_ids, short answer)
FACTOID: list[tuple[str, str, list[str], str]] = [
    ("f-001", "By what percentage did empagliflozin reduce clinical events of hyperuricaemia such as acute gout in patients with heart failure and preserved ejection fraction?", ["4b0d1b6622b627e8"], "38% (HR 0.62, 95% CI 0.51-0.76)"),
    ("f-002", "What serum uric acid levels define hyperuricaemia in men and in women in the EMPEROR-Preserved trial analysis?", ["4b0d1b6622b627e8"], "Above 7.0 mg/dL in men and above 5.7 mg/dL in women"),
    ("f-003", "What LDL cholesterol level makes an adult aged 21 or over statin-eligible under the 2013 ACC/AHA cholesterol guideline regardless of other risk factors?", ["12bae62eb8955594"], "LDL-C of 190 mg/dL (4.9 mmol/L) or higher"),
    ("f-004", "What percentage of statin-eligible US adults with diabetes were actually taking a statin in NHANES 2011-2012?", ["12bae62eb8955594"], "43.2%"),
    ("f-005", "Does the choice between an ACE inhibitor and an angiotensin receptor blocker change the risk of progression to end-stage renal disease in chronic kidney disease?", ["77f70a57e00f15bb"], "No significant difference (HR 1.31, 95% CI 0.37-4.66)"),
    ("f-006", "Which statin regimens achieved significantly higher cholesterol goal attainment than simvastatin 40 mg daily?", ["ab7294f4c46e71b4"], "Atorvastatin 40 mg/day (RR 1.15) and rosuvastatin 10 mg/day (RR 1.13)"),
    ("f-007", "What was the target dose of candesartan in the CHARM-Preserved trial?", ["f6bfb9fba1b806ae"], "32 mg once daily"),
    ("f-008", "Did candesartan reduce cardiovascular death in patients with chronic heart failure and left ventricular ejection fraction above 40 percent?", ["f6bfb9fba1b806ae"], "No - cardiovascular death did not differ (170 vs 170)"),
    ("f-009", "How many randomised controlled trials and participants were included in the network meta-analysis of empiric antibiotics for moderate-to-severe community-acquired pneumonia?", ["fd8a9a300520795f"], "143 RCTs involving 29,157 participants"),
    ("f-010", "What proportion of hospitalised heart failure patients with available ejection fraction data had preserved ejection fraction in the Olmsted County study?", ["4926ac78f1a27aa9"], "47 percent"),
    ("f-011", "How many abnormally elevated laboratory values does the German National Disease Management Guideline now require to diagnose type 2 diabetes?", ["a4e3cdca21cbbc2b"], "At least two"),
    ("f-012", "What fasting plasma glucose threshold is used to diagnose type 2 diabetes in the German National Disease Management Guideline?", ["a4e3cdca21cbbc2b"], "126 mg/dL (7.0 mmol/L) or higher"),
    ("f-013", "How does mean length of hospital stay compare between oral and parenteral antibiotic therapy for community-acquired pneumonia?", ["0d09bd942163bc65"], "Shorter with oral therapy: 6.1 days vs 7.8 days"),
    ("f-014", "Which antibiotic was most frequently used in patients with Legionella pneumophila pneumonia in the Palermo case-control study?", ["30b5b49e8e6e56c8"], "Levofloxacin"),
    ("f-015", "Which index was used to measure insulin resistance in the UK Biobank analysis of progression from fatty liver disease to cardiovascular-kidney-metabolic disease?", ["4f1e9290b446587a"], "The TyG index"),
    ("f-016", "What happened to peak oxygen consumption when beta-blockers were withdrawn in patients with preserved ejection fraction heart failure and chronotropic incompetence?", ["ce37cf524d6561af"], "It increased from 12.2 to 14.3 mL/kg/min, a rise of 2.1"),
    ("f-017", "What was the effect of finerenone on cardiovascular death or heart failure hospitalisation in patients with mildly reduced or preserved ejection fraction?", ["d3b42767bac562d7"], "Reduced it (HR 0.87, 95% CI 0.78-0.96)"),
    ("f-018", "By how much does each doubling of NT-proBNP increase the risk of the primary outcome in heart failure with preserved ejection fraction?", ["f572a4ed03c23502"], "37% relative increase (HR 1.37)"),
    ("f-019", "What modified H2FPEF score threshold identified a high likelihood of preserved ejection fraction heart failure in the CABANA trial analysis?", ["2ee45773e112db8d"], "A score of 6 or higher"),
    ("f-020", "By how much does controlling all four modifiable risk factors reduce chronic kidney disease risk in patients with hypertension?", ["83140d3ba6a2e125"], "42% (HR 0.58, 95% CI 0.52-0.63)"),
    ("f-021", "Was antibiotic therapy continued beyond 7 days associated with a survival benefit in severe community-acquired pneumonia?", ["16885ac57ed84889"], "No - no survival benefit, and longer ICU and hospital stay"),
    ("f-022", "By what percentage is adopting a plant-based diet associated with lower incidence of chronic kidney disease?", ["2d0aac2862506b57"], "26% lower (OR 0.75, 95% CI 0.65-0.86)"),
    ("f-023", "Do patients hospitalised with possible pneumonia and a positive respiratory virus test have better outcomes with 5 to 7 days of antibacterials than with 0 to 2 days?", ["336d8e69d085d486"], "No significant differences in any outcome"),
    ("f-024", "How did in-hospital mortality compare between azithromycin and doxycycline when each was combined with a beta-lactam in hospitalised pneumonia patients?", ["45ec68673f629617"], "Lower with azithromycin (OR 0.71, 95% CI 0.56-0.9)"),
    ("f-025", "What effect did guideline-concordant antibiotic therapy have on one-year cardiovascular death risk in older patients surviving hospitalisation for pneumonia?", ["50fa81915dc8615e"], "Almost 50% reduction (HR 0.53, 95% CI 0.34-0.80)"),
    ("f-026", "What effect did beta-blocker therapy have on all-cause mortality in preserved ejection fraction heart failure in pooled observational cohort studies?", ["59dd06714046f8ec"], "19% reduction (OR 0.81, 95% CI 0.65-0.99)"),
    ("f-027", "What was the effect of catheter ablation versus standard medical therapy on the composite of all-cause mortality and heart failure hospitalisation in preserved ejection fraction heart failure with atrial fibrillation?", ["5cdb791d5936eff9"], "Lower risk with ablation (HR 0.61, 95% CI 0.43-0.85)"),
    ("f-028", "How is chronic kidney disease defined in terms of glomerular filtration rate and albuminuria, and how long must the abnormality persist?", ["62814962a28513a7"], "GFR below 60 mL/min/1.73 m2 or albuminuria of 30 mg per 24 hours or more, for more than 3 months"),
    ("f-029", "At what estimated GFR or albuminuria level should a patient with chronic kidney disease be referred promptly to a nephrologist?", ["62814962a28513a7"], "eGFR below 30 mL/min/1.73 m2, albuminuria 300 mg per 24 hours or more, or rapid eGFR decline"),
    ("f-030", "What percentage of Thai patients with type 2 diabetes and no cardiovascular disease achieved an HbA1c below 6.5 percent?", ["8030d1b30a12f034"], "28%"),
]

MULTI_HOP: list[tuple[str, str, list[str], str]] = [
    ("m-001", "In preserved ejection fraction heart failure, what effect do beta-blockers have on exercise capacity, and what effect do they have on all-cause mortality?", ["ce37cf524d6561af", "59dd06714046f8ec"], "Withdrawal improved peak VO2 by 2.1 mL/kg/min; pooled observational data show 19% lower all-cause mortality with beta-blockers"),
    ("m-002", "What do a randomised trial analysis and a meta-analysis each report about catheter ablation for atrial fibrillation in patients with preserved ejection fraction heart failure?", ["2ee45773e112db8d", "5cdb791d5936eff9"], "CABANA: HR 0.82 for cardiovascular hospitalisation or death in high HFpEF likelihood; meta-analysis: HR 0.61 for the composite of mortality and HF hospitalisation"),
    ("m-003", "In community-acquired pneumonia treated with a beta-lactam, how does adding a macrolide affect mortality, and how does azithromycin compare with doxycycline?", ["16885ac57ed84889", "45ec68673f629617"], "Macrolide combination reduced hospital mortality (OR 0.17); azithromycin had lower in-hospital mortality than doxycycline (OR 0.71)"),
    ("m-004", "How is chronic kidney disease defined, and does the choice between ACE inhibitors and angiotensin receptor blockers change progression to end-stage renal disease?", ["62814962a28513a7", "77f70a57e00f15bb"], "GFR below 60 or albuminuria 30 mg/24h for over 3 months; ACEI and ARB efficacy was comparable (HR 1.31, ns)"),
    ("m-005", "How do reduced kidney function and higher body mass index each affect interpretation of NT-proBNP thresholds in preserved ejection fraction heart failure?", ["f572a4ed03c23502", "694de914cf5dc46f"], "Low eGFR: same NT-proBNP predicts higher absolute risk; high BMI: current thresholds underestimate risk, so lower cutoffs may be needed"),
    ("m-006", "Which adults are statin-eligible under ACC/AHA criteria, and which statin regimens best achieve cholesterol goals?", ["12bae62eb8955594", "ab7294f4c46e71b4"], "Four groups: ASCVD, LDL-C 190+, diabetes aged 40-75, or 10-year risk 7.5%+; atorvastatin 40 mg and rosuvastatin 10 mg outperformed simvastatin 40 mg"),
    ("m-007", "What HbA1c threshold diagnoses type 2 diabetes, and what proportion of Thai patients with type 2 diabetes achieved an HbA1c below that level?", ["a4e3cdca21cbbc2b", "8030d1b30a12f034"], "HbA1c of 6.5% (48 mmol/mol) or higher diagnoses diabetes; only 28% of Thai patients were below 6.5%"),
    ("m-008", "How much does controlling multiple modifiable risk factors in hypertension reduce chronic kidney disease risk, and how much does adopting a plant-based diet reduce it?", ["83140d3ba6a2e125", "2d0aac2862506b57"], "Controlling all four factors: 42% lower risk (HR 0.58); plant-based diet: 26% lower incidence (OR 0.75)"),
]

AGGREGATION: list[tuple[str, str, list[str], str]] = [
    ("a-001", "Which drug classes have been evaluated as treatments for heart failure with preserved ejection fraction?", ["d3b42767bac562d7", "f6bfb9fba1b806ae", "4b0d1b6622b627e8", "59dd06714046f8ec"], "Mineralocorticoid receptor antagonists (finerenone), angiotensin receptor blockers (candesartan), SGLT2 inhibitors (empagliflozin), and beta-blockers"),
    ("a-002", "Which different antibiotic strategies have been compared for treating community-acquired pneumonia in hospitalised adults?", ["fd8a9a300520795f", "45ec68673f629617", "16885ac57ed84889", "0d09bd942163bc65", "336d8e69d085d486"], "Empiric regimen classes, azithromycin vs doxycycline with beta-lactams, beta-lactam plus macrolide combination, oral vs parenteral therapy, and short vs longer courses with viral co-infection"),
    ("a-003", "What interventions have been shown to slow the progression of chronic kidney disease?", ["77f70a57e00f15bb", "2d0aac2862506b57", "83140d3ba6a2e125"], "ACE inhibitors or ARBs, adopting a plant-based diet, and multifactorial control of blood pressure, LDL, glucose and BMI"),
    ("a-004", "Which heart failure studies are identified by a ClinicalTrials.gov NCT registration number?", ["4b0d1b6622b627e8", "ce37cf524d6561af", "2ee45773e112db8d"], "EMPEROR-Preserved (NCT03057951), PRESERVE-HR (NCT03871803), and CABANA (NCT00911508)"),
    ("a-005", "How well are cholesterol and glycaemic treatment targets actually achieved in routine clinical practice?", ["12bae62eb8955594", "ab7294f4c46e71b4", "8030d1b30a12f034"], "Poorly: many statin-eligible US adults are untreated or above goal, statin choice affects attainment, and only 28% of Thai diabetes patients reached HbA1c below 6.5%"),
]

# Adjacent enough that retrieval will confidently return passages, but not answerable
# from any of them. This is what actually tests refusal.
UNANSWERABLE: list[tuple[str, str]] = [
    ("u-001", "What insulin infusion rate is recommended for treating diabetic ketoacidosis in adults?"),
    ("u-002", "Which direct oral anticoagulant carries the lowest risk of gastrointestinal bleeding in atrial fibrillation?"),
    ("u-003", "What is the target INR range for a patient with a mechanical mitral valve replacement?"),
    ("u-004", "How long should dual antiplatelet therapy be continued after placement of a drug-eluting coronary stent?"),
    ("u-005", "What vancomycin dose is recommended for methicillin-resistant Staphylococcus aureus pneumonia?"),
    ("u-006", "Which potassium binder is preferred for treating hyperkalaemia caused by ACE inhibitor therapy?"),
    ("u-007", "At what estimated GFR should metformin be discontinued in patients with type 2 diabetes?"),
    ("u-008", "At what estimated GFR should maintenance dialysis be initiated in chronic kidney disease?"),
    ("u-009", "Which pneumococcal vaccine schedule is recommended for adults over 65 to prevent pneumonia?"),
    ("u-010", "What is the five-year survival rate after heart transplantation for end-stage heart failure?"),
    ("u-011", "How many grams per day of dietary sodium restriction are recommended in chronic heart failure?"),
    ("u-012", "Which SGLT2 inhibitor is preferred for patients with type 1 diabetes?"),
    ("u-013", "What is the sensitivity of procalcitonin for distinguishing bacterial from viral pneumonia?"),
    ("u-014", "What is the recommended duration of anticoagulation after a first unprovoked pulmonary embolism?"),
]


def main() -> int:
    cfg = Config.load()
    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=False)

    items: list[GoldenItem] = []
    missing: list[str] = []

    def add(qid: str, question: str, qtype: str, gold: list[str], answer: str | None) -> None:
        # Fail loudly on a stale id rather than silently writing an unanswerable label:
        # a gold id that does not exist scores 0 recall forever and looks like a
        # retrieval failure.
        for cid in gold:
            if index.get(cid) is None:
                missing.append(f"{qid} -> {cid}")
        items.append(
            GoldenItem(
                qid=qid,
                question=question,
                question_type=qtype,  # type: ignore[arg-type]
                gold_chunk_ids=gold,
                answer=answer,
                provenance={"source": "hand-written", "corpus_fingerprint": index.fingerprint},
            )
        )

    for qid, q, gold, ans in FACTOID:
        add(qid, q, "factoid", gold, ans)
    for qid, q, gold, ans in MULTI_HOP:
        add(qid, q, "multi_hop", gold, ans)
    for qid, q, gold, ans in AGGREGATION:
        add(qid, q, "aggregation", gold, ans)
    for qid, q in UNANSWERABLE:
        add(qid, q, "unanswerable", [], None)

    if missing:
        print("ERROR: gold chunk ids not found in the index:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nRe-run `ragmed index`, or update the ids in this file.", file=sys.stderr)
        return 1

    # Multi-hop must genuinely span documents, or it is a factoid wearing a costume.
    for it in items:
        if it.question_type == "multi_hop":
            docs = {index.get(c).doc_id for c in it.gold_chunk_ids}  # type: ignore[union-attr]
            if len(docs) < 2:
                print(f"ERROR: {it.qid} is labelled multi_hop but all gold chunks share a document",
                      file=sys.stderr)
                return 1

    out = cfg.eval.golden_set
    save_golden(out, items)

    counts: dict[str, int] = {}
    for it in items:
        counts[it.question_type] = counts.get(it.question_type, 0) + 1
    print(f"wrote {len(items)} questions to {out}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<14} {v}")
    print(f"  answerable     {sum(1 for i in items if i.is_answerable)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
