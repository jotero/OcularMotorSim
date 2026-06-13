# SFN Abstract — Oculomotor Simulator

> Draft for Society for Neuroscience annual meeting. Edit freely.
> SFN abstract body limit is **2,300 characters including spaces** (title and author block excluded).
> Pick one theme/topic below and delete the others.

---

## Title

A virtual eye-movement laboratory: a whole-system model of human oculomotor control for teaching and synthetic-data generation

## Authors / Affiliation

[First Last]¹, [Coauthors]¹; ¹Oculomotor Laboratory, University of California, Berkeley, CA

## Suggested theme / topic category

- **Theme F: Cognition and Behavior** → Topic: *Eye Movements* (primary recommendation)
- Alternative — **Theme D: Sensory Systems** → Topic: *Vestibular System* (if framing toward VOR / velocity storage)
- Keywords: oculomotor; eye movements; computational model; vestibulo-ocular reflex; medical education; synthetic data

---

## Abstract

Eye movements are a sensitive, quantifiable window into neurological disease: focal brainstem, cerebellar, and vestibular lesions produce characteristic signs that clinicians use to localize pathology at the bedside. Yet these signs can be hard to teach and quantify, and most computational models capture a single behavior—saccades, the vestibulo-ocular reflex (VOR), or vergence—in isolation, limiting their use for teaching or for generating data to constrain automated-diagnosis pipelines. We present a virtual eye-movement laboratory: a unified simulator of the human oculomotor system that reproduces the full repertoire of eye movements within a single architecture and parameter set, in which focal lesions reproduce clinical syndromes.

Rather than fitting each behavior separately, a single biophysically grounded parameter set governs the whole system—from the vestibular and visual periphery through the brainstem, neural integrator, and cerebellum to a binocular plant—so that reflexive stabilization, voluntary gaze shifts, and binocular alignment all emerge from one circuit and quantitatively match the classic benchmarks of each. Because every component maps onto an identifiable neural structure, lesioning a pathway reproduces the corresponding clinical syndrome, from internuclear ophthalmoplegia to gaze-evoked nystagmus.

Every parameter corresponds to a specific physiological quantity—time constants, gains, and lesion strengths—so the model is mechanistically interpretable and can be fit to recorded eye movements. Because any stimulus can be applied to any healthy or lesioned subject, the laboratory serves two roles. First, the model is distributed with a companion website on which a user simply types a scenario in plain language (e.g., "left vestibular neuritis during a head-impulse test"); the site configures the corresponding experiment and patient, runs the simulation, and returns annotated figures, letting students and clinicians explore normal and pathological eye movements without writing code. Second, it generates synthetic data: responses simulated under known parameters and noise yield labeled datasets for validating parameter-recovery and disease-classification methods before use on patient data. Code, benchmarks, and web tool are openly available.
