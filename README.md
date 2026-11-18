# Hebrew-Gendered-Nikud

A Hebrew diacritizer (nikud model) with **gender as a first-class input**. You pass unpointed Hebrew and a gender target (`m`, `f` or `none`), and the model points the words whose vowels depend on gender to match that target.

- Base: [`dicta-il/dictabert-large-char-menaked`](https://huggingface.co/dicta-il/dictabert-large-char-menaked), revision `4dfc20b94565d7868e1c8c00bf039d38c071a062`, CC-BY-4.0, by Dicta.
- Hugging Face repo: [`Hebrish/Hebrew-Gendered-Nikud`](https://huggingface.co/Hebrish/Hebrew-Gendered-Nikud), created 2026-11-18. It is private and still an **unmodified server-side copy** of that revision. It becomes public with the first trained version.
- Status (2026-11-18): planning. Nothing has been trained yet.

## Why

The upstream model's `predict(sentences, tokenizer, mark_matres_lectionis=None)` takes no gender control. Checked in `BertForDiacritization.py` on 2026-11-18: each character gets the argmax of `nikud_logits` / `shin_logits`. So when a word is ambiguous, the model guesses the gender from the surrounding text, and in practice it guesses masculine.

The main use is generation: writing Hebrew sentences for a given gender that people then read aloud (ASR training data, TTS).

## Where gender lives only in the nikud

In most Hebrew, gender is already in the letters: הולך/הולכת, אתה/את, אשתו/אשתה. A nikud model cannot change letters and does not need to. The control matters only where the **unpointed letters are identical**. In practice those are second-person singular forms, i.e. the gender of the person **addressed**:

| Unpointed | Masculine | Feminine |
|---|---|---|
| לך | לְךָ | לָךְ |
| שלך | שֶׁלְּךָ | שֶׁלָּךְ |
| אותך | אוֹתְךָ | אוֹתָךְ |
| בך / ממך / עליך / אליך | בְּךָ / מִמְּךָ / עָלֶיךָ / אֵלֶיךָ | בָּךְ / מִמֵּךְ / עָלַיִךְ / אֵלַיִךְ |
| noun + ך (ספרך) | סִפְרְךָ | סִפְרֵךְ |
| past tense 2nd person (הלכת, עשית) | הָלַכְתָּ, עָשִׂיתָ | הָלַכְתְּ, עָשִׂית |

The speaker's own gender (first person) is written in the letters, so the generator handles it, not this model. Not yet decided: whether to name the input `addressee_gender` and add a separate `speaker_gender` later, or keep one `gender` field.

## Plan

1. **Baseline probe.** Run the upstream model on about 50 sentences built around the forms in the table, some with context that signals gender and some neutral. Measure how often it picks masculine, feminine, or something inconsistent. Runs on Modal, never on a laptop.
2. **Data.** Start from pointed modern Hebrew: Dicta output on unambiguous text, plus hand-checked rows. For each sentence that contains a form from the table, write both gendered versions with a rule-based vowel flip over that closed set of patterns. Gender-neutral sentences keep the same target under every tag, so the model learns that the tag changes only the ambiguous points.
3. **Conditioning.** Prepend a control token (`[M]`, `[F]`, `[N]`) to the character input. Fine-tune the existing nikud and shin heads; the architecture does not change. `predict()` gets a `gender=` argument.
4. **Evaluation.** Accuracy on the ambiguous forms under each tag, and no loss of whole-sentence nikud accuracy against the upstream model on neutral text.
5. **Later, the reverse direction.** Classify whether a pointed text is addressed to a man or a woman. That is a cheap byproduct of the same rules.

## Licence

The derivative weights will be CC-BY-4.0 with attribution to Dicta (DictaBERT-large-char-menaked), the same licence as upstream.
