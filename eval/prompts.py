# prompts.py
# -*- coding: utf-8 -*-

VISUAL_CONSISTENCY_TEMPLATE="""
You are a professional digital artist and image evaluation specialist.
You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.
4. **Editing Object**:the object or objects that are intended to be modified.
5. **Object description**:the detailed description of **Editing Object**, used for localization and comparison.
Your Objective:
Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively 
on elements that are NOT specified for change in the instruction**. That is, you should only consider whether 
all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly 
required by the instruction.
## Evaluation Scale (1 to 5):
You will assign a **consistency_score** according to the following rules:
- **5 Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **4 Minor Inconsistency**: Only one very small, non-instruction detail is different (e.g., a tiny accessory, a 
subtle shadow, or a minor background artifact).
- **3 Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a 
shifted object, or a visible background alteration).
- **2 Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1 Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, 
gender, or overall scene layout).
## Guidance:
- First, **identify all elements that the instruction explicitly allows or requires to be changed**. 
To identify what is explicitly required, first locate the **Editing Object** with its **Object description**
Exclude these from your consistency check;then verify that every other pixel and element remains unchanged between the original and edited images.
- For all other elements (e.g., facial features, clothing, background, object positions, colors, lighting, scene 
composition,especially image color tone,brightness, contrast, etc.), **compare Image B to Image A** and check if they remain visually identical.
- If you observe any change in a non-instruction element, note it and consider its impact on the score.
- If the instruction is vague or ambiguous, make a best-effort factual inference about which elements are 
intended to change, and treat all others as non-instruction elements.
## Note:
- **Do not penalize changes that are required by the instruction.**
- **Do not reward or penalize the quality or correctness of the instructed change itself** (that is evaluated 
separately).
- If the edited image introduces new artifacts, objects, or changes to non-instruction elements, this should lower 
the consistency score.
## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
First, clearly explain your comparison process: list each major non-instruction element and state whether it is 
consistent (unchanged) or inconsistent (changed), with brief reasoning.
Then, provide your evaluation in the following JSON format:
{{
"reasoning": **Compared to original image**, [list of non-instruction elements that changed or remained the 
same] **in the edited image**. 
"consistency_score": X
}}

"""

VISUAL_QUALITY_TEMPLATE="""
You are a professional digital artist and image evaluation specialist.
You will be given:
- **Image A**: a single AI-generated image.
## Objective:
Your task is to **evaluate the perceptual quality** of the image, focusing on:
- **Structural and semantic coherence**
- **Natural appearance**
- **Absence of generation artifacts**
You must **not penalize low resolution or moderate softness** unless it introduces semantic ambiguity or 
visually degrading effects.
## Evaluation Scale (1 to 5):
You will assign a **quality_score** with the following rule:
- **5 Excellent Quality**: All aspects are visually coherent, natural, and free from noticeable artifacts. 
Structure, layout, and textures are accurate and consistent.
- **4 Minor Issues**: One small imperfection (e.g., slight texture blending, minor lighting inconsistency).
- **3 Noticeable Artifacts**: One or two clear visual flaws or semantic problems (e.g., extra fingers, minor 
duplication, slight distortion).
- **2 Structural Degradation**: Multiple distracting errors (e.g., melted hands, warped shapes, unreadable text).
- **1 Severe Errors**: Major structural failures or hallucinations (e.g., broken anatomy, garbled symbols).
## Guidance:
Check the following visual aspects and mark them as ✔ (satisfactory) or ✘ (problematic):
- Structural coherence (e.g., correct anatomy, object shapes, legible text)
- Naturalness (lighting, perspective, shadow logic)
- Artifact-free (no duplication, ghosting, watermarks)
- Texture fidelity (clothing, hair, surfaces not melted or corrupted)
- Optional: Sharpness (only penalize if blur causes semantic loss)
✔ The more checks, the higher the score.
Example
"reasoning": "Structural coherence: ✔, Natural appearance: ✔, Artifacts: ✔, Texture fidelity: ✘ (fabric 
partially deformed).",
"quality_score": 4
## Output Format:
After evaluation, provide your score and concise reasoning using the following JSON format:
{{
"reasoning": XXX,
"quality_score": X,
}}
"""

INSTRUCTION_FOLLOWING_TEMPLATE="""
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AIgenerated image(s) based on given rules. 
You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a hypothetical prompt describing how Image A should be transformed or imagined into Image B.
4. **Editing Object**:the object or objects that are intended to be modified.
5. **Object description**:the detailed description of **Editing Object**, used for localization and comparison.
Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the 
presence and correctness of the specified changes**. 
You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
**Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)
## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, state) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?To identify the correct object, first locate the **Editing Object** with its **Object description**
- Was the requested attribute (e.g., size, color, state) modified as intended?
- Is the degree of modification accurate (e.g., “match size”, “an hour later,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.
## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present 
and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 
Example: 
Instruction: Imagine what the apple will look like in a month.
{{
"instruction_score": 3,
"reasoning": "
1. Detecting differences: 
In the original image, the apple shows a bright red color and a smooth surface. However, in the processed image, some black spots appear on the surface of the apple. 
2. Expected visual description: 
The originally shiny skin of the fruit would become dull and brown, with dark spots and blemishes appearing on it. As the water content decreases, the fruit will become wrinkled and shrink in size, and the flesh will become soft and spongy, and may even collapse in some areas. 
3. Explanation:
Comparison: This instruction requires imagining the appearance of the apple after one month. This editing adds black spots to the apple's surface, which to some extent meets the requirements of the instruction, but the apple does not undergo dehydration or wrinkling changes. The core concept was attempted, but it was not fully achieved. 
4. Decision: 
Since only part of the apple's appearance reached the required level, this should be counted as 3 cases that partially meet the requirements."
}}
## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
Look at the input again, provide the evaluation score and the explanation in the following JSON format:
{{
"instruction_score": X,
"reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
}}
"""

KNOWLEDGE_PLAUSIBILITY_TEMPLATE="""
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI generated image(s) based on given rules. 
You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.
4. **editing_mode**: the fundamental editing regime that must be observed throughout the image-editing process; it dictates the governing condition under which all operations are performed—e.g., break, time, temperature, stretch, etc.
5. **Editing Object**:the object or objects that are intended to be modified.
6. **Object description**:the detailed description of **Editing Object**, used for localization and comparison.
## Objective
You must provide **scores** for the **edited image**:
- **Knowledge Score**: Given the instruction and original image, does the edited image reflect what should realistically happen based on the mode?

##  Knowledge Plausibility 
Your Objective:
Evaluate whether the edited image, after applying the instruction to the original image, accurately reflects the real-world behavior 
described in the provided mode.
You must:
**Ground your reasoning in the Real-World Knowledge Explanation based on the provided mode.**
Focus only on whether the resulting image makes logical sense based on **physical, chemical, biological, or commonsense understanding**.
**Not penalize issues unrelated to knowledge** (e.g., visual polish or stylistic artifacts)
## Reasoning Steps:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color) In this
step, you don't have to use information from the editing instruction
**2. Extract Knowledge Expectation**: What visual outcome is expected if the instruction is applied, based on the provided 
mode?
**3. Knowledge Match**: 
Compare the visual changes identified in Step 1 to the expected outcome in Step 2:
- Was the correct object modified (not replaced)?To identify the correct object, first locate the **Editing Object** with its **Object description**
- Do the edits visually and logically match the real-world behavior?
- Is the cause-effect relationship shown correctly?
- Are key physical/chemical/biological phenomena depicted correctly?
**4. Decision**: Assign a knowledge_score from 1 to 5
### Evaluation Scale (1 to 5):
- **5 Fully Plausible**: All visual elements follow real-world logic and match the explanation exactly.
- **4 Minor Implausibility**: One small deviation from expected real-world behavior.
- **3 Noticeable Implausibility**: One clear conflict with domain knowledge or the explanation.
- **2 Major Implausibility**: Multiple serious violations of the real-world logic.
- **1 Completely Implausible**: The image contradicts fundamental facts or ignores the explanation entirely.
If instruction is not followed (score ≤ 2), assign `knowledge_score = 1` and note: *"Instruction failure ⇒ knowledge invalid."*
### Example 1: What if the rose is placed on the table for a month?
**Editing Instruction**: What if the rose is placed in the vase for a month? 
**Editing mode**: time.
- **Compared to original image**, the rose is dry, droopy, and faded.
→ **Expected Caption**: The rose is dry, droopy, and faded.
"knowledge_score": 5,
"reasoning": "✔ The rose is dry, droopy, and faded."
### Example 2: The glass breaks.
**Editing Instruction**: The glass breaks under the server pressure. 
**Editing mode**: break.
- ✔ **Compared to original image**, Tiny cracks spread across the glass surface. 
- ✘ The glass only slightly breaks instead of completely breaking, contradicting real-world behavior under server pressure.
→ **Expected Caption**: The glass shattered completely into small sharp pieces.
"knowledge_score": 3,
"reasoning": "✘ The degree of breakage is too small, contradicting real-world behavior under server pressure."
## Input
**Original Image**
**Edited Image**
**Editing Instruction**: {instruct}
**Editing mode**：{mode}
## Output Format
Provide both scores and clear reasoning in the following JSON format:
{{
"knowledge_score": X,
"knowledge_reasoning": 1. Detect Difference 2. Expected Knowledge Expectation 3. Knowledge Match 4. Decision
}}

"""