# Hebrew-Gendered-Nikud

![Status: idea / future](https://img.shields.io/badge/status-idea%20%2F%20future-lightgrey) ![Stage: 2 of 2](https://img.shields.io/badge/pipeline-stage%202%20of%202-informational)

A Hebrew diacritizer (nikud model) with **speaker gender as a first-class input**. You pass unpointed Hebrew and the speaker's gender (`m`, `f` or `none`), and the model points the words whose vowels depend on that gender to match it.

- Base: [`dicta-il/dictabert-large-char-menaked`](https://huggingface.co/dicta-il/dictabert-large-char-menaked), revision `4dfc20b94565d7868e1c8c00bf039d38c071a062`, CC-BY-4.0, by Dicta.
- Hugging Face repo: [`Hebrish/Hebrew-Gendered-Nikud`](https://huggingface.co/Hebrish/Hebrew-Gendered-Nikud), created 2026-11-18. It is private and still an **unmodified server-side copy** of that revision. It becomes public with the first trained version.
- Status (2026-11-18): idea, for the future. Nothing has been trained. Work starts after stage 1, [Hebrew-Gendered-Sentence-Creator](https://github.com/Hebrish-ASR/Hebrew-Gendered-Sentence-Creator).

## Why

The upstream model's `predict(sentences, tokenizer, mark_matres_lectionis=None)` takes no gender control. Checked in `BertForDiacritization.py` on 2026-11-18: each character gets the argmax of `nikud_logits` / `shin_logits`. So when a word is ambiguous, the model guesses the gender from the surrounding text. Which way it leans is not measured yet (plan step 1); masculine is the assumption.

The main use is generation: writing Hebrew sentences for a given gender that people then read aloud (ASR training data, TTS).

## The pipeline: why two models

Hebrew marks the speaker's gender in two ways. **In the letters** (אני הולך / אני הולכת) the sentence has to be *written* for the right speaker; a nikud model only adds vowels to the letters it is given and cannot do that. That is stage 1, [Hebrew-Gendered-Sentence-Creator](https://github.com/Hebrish-ASR/Hebrew-Gendered-Sentence-Creator). **Only in the vowels** (אני רוצה: רוֹצֶה / רוֹצָה) the letters are already right and only nikud shows the gender. That is this repo, stage 2.

```
term + context + speaker_gender
  -> [1] Hebrew-Gendered-Sentence-Creator -> unpointed Hebrew, letters right for the speaker
  -> [2] Hebrew-Gendered-Nikud (gender=m|f) -> pointed Hebrew, vowels right for the speaker
```

The stages are separate models because LLMs are weak at writing nikud and dedicated nikud models are strong at it. Separate models can also be tested and replaced one at a time. This model also works on any Hebrew text, not only stage 1 output.

## Target: the speaker's gender (Daniel, 2026-11-18)

The control is the gender of the **speaker** (first person), not of the person addressed. Most speaker gender is written in the letters (אני הולך / אני הולכת, אני שמח / אני שמחה). A diacritizer cannot change letters, so those forms are the sentence generator's job; this model points whatever letters it is given.

What the model must decide is the set where the **unpointed letters are identical for both genders** and only the vowels change. That is mainly present tense of ל"ה verbs and adjectives ending in ה:

| Unpointed | Masculine speaker | Feminine speaker |
|---|---|---|
| אני רוצה | אֲנִי רוֹצֶה | אֲנִי רוֹצָה |
| אני קונה / עושה / רואה / שותה / בונה | קוֹנֶה / עוֹשֶׂה / רוֹאֶה / שׁוֹתֶה / בּוֹנֶה | קוֹנָה / עוֹשָׂה / רוֹאָה / שׁוֹתָה / בּוֹנָה |
| אני מנסה / מקווה / מצפה / מפנה | מְנַסֶּה / מְקַוֶּה / מְצַפֶּה / מְפַנֶּה | מְנַסָּה / מְקַוָּה / מְצַפָּה / מְפַנָּה |
| (אני) יפה / קשה / רזה | יָפֶה / קָשֶׁה / רָזֶה | יָפָה / קָשָׁה / רָזָה |

The class is the closed ֶה ↔ ָה alternation at the end of a word whose subject is the speaker. That makes rule-based flipping and checking straightforward. The model still has to work out whether a given ה-final word refers to the speaker (אני רוצה) or to something else (הוא רוצה, the speaker is irrelevant).

Second-person forms, where only the vowels show who is addressed (לְךָ/לָךְ, הָלַכְתָּ/הָלַכְתְּ), are a possible second control (`addressee_gender`) later. They are out of scope for v1.

## Prior work (searched 2026-11-18)

Nothing found for Hebrew does either job with speaker gender as an input: not generating text for a given speaker gender, and not diacritizing for one.

- **Hebrew nikud models predict gender from context only.** [D-Nikud](https://arxiv.org/abs/2402.00075) (2024) lists "gender" as an emphasis and reports a "Female Texts" test (Table 4: 97.99 DEC vs Nakdimon 96.92). Its `predict` has no gender argument (repo [NadavShaked/D_Nikud](https://github.com/NadavShaked/D_Nikud)), and it is unclear whether that test set was released. DictaBERT-menaked has no gender argument either (checked in code). No public documentation of a gender option in [Nakdan](https://nakdanpro.dicta.org.il) was found; that is unconfirmed, not ruled out.
- **The closest template is Arabic.** Habash et al. 2019 ([W19-3822](https://aclanthology.org/W19-3822/)) did first-person-singular gender identification and reinflection. The [Arabic Parallel Gender Corpus](https://aclanthology.org/2022.lrec-1.199) v1 labels OpenSubtitles sentences M/F/B (ambiguous) with opposite-gender twins; v2 adds second person (about 590K words). The neural rewriter [User-Centric Gender Rewriting](https://arxiv.org/abs/2205.02211) (2022) controls speaker and addressee gender separately and rewrites only the ambiguous parts. This is the design to copy for a Hebrew rewriter.
- **Multilingual LLM benchmark.** [MORPHOGEN](https://arxiv.org/abs/2604.18914) (2026) tests first-person gender rewriting in French, Arabic and Hindi, not Hebrew. Its main failure mode is relevant here: the LLM gets the speaker right but also flips the gender of *other* people in the sentence.
- **Gender augmentation.** [Generating Gender Augmented Data for NLP](https://aclanthology.org/2021.gebnlp-1.11.pdf) (2021) names Hebrew among the target languages. It is rule-based and not specific to first person.

## Plan

1. **Baseline probe.** Run the upstream model on about 50 first-person sentences built around the ל"ה and ה-final forms in the table, some with a gender cue elsewhere in the sentence (אני רוצה, אמרתי לבעלי / לאשתי) and some neutral. Measure how often it picks masculine, feminine, or something inconsistent. Runs on Modal, never on a laptop.
2. **Data.** Start from pointed modern Hebrew: Dicta output on unambiguous text, plus hand-checked rows. For each sentence where a form from the table refers to the speaker, write both gendered versions with a rule-based ֶה ↔ ָה flip. Sentences where the speaker's gender is fixed in the letters (הולכת) teach the model to follow the letters when they decide. Gender-neutral sentences keep the same target under every tag, so the model learns that the tag changes only the ambiguous points.
3. **Conditioning.** Prepend a control token (`[M]`, `[F]`, `[N]`) to the character input. Fine-tune the existing nikud and shin heads; the architecture does not change. `predict()` gets a `gender=` argument.
4. **Evaluation.** Accuracy on the ambiguous forms under each tag, and no loss of whole-sentence nikud accuracy against the upstream model on neutral text.
5. **Later, the reverse direction.** Classify whether a pointed text was written for a male or a female speaker. That is a cheap byproduct of the same rules.

## Licence

The derivative weights will be CC-BY-4.0 with attribution to Dicta (DictaBERT-large-char-menaked), the same licence as upstream.
