from dataclasses import dataclass
from enum import StrEnum

from app.custom_parser import MY_PARSER
from langchain_core.prompts import PromptTemplate

coherence_template = """You are the judge evaluating coherence/logic when you receive the debate topic and the content of the discussion. Please score it out of 100 points and submit your assessment.

## Rule
Coherence/Logic:
- When logical flow is disrupted or there's inconsistency between claims and evidence.
- When claims change inconsistently or are repeated without coherence.
- When there are unnatural or difficult-to-understand points logically.

Below are specific examples that could result in deductions. Please consider these when evaluating.

1. When the logical flow is disrupted or there are contradictions between claims and evidence:
   - Example: Team A argues that maximizing a company's profits is essential, but their justification is that not maximizing profits could have negative societal impacts. This is a contradictory claim where the logical flow between maximizing profits and considering social responsibility is disrupted.
2. When arguments change inconsistently or are repeated without coherence:
   - Example: Team B initially emphasizes a company's social responsibility but later shifts to arguing that profit maximization is more important. Such changes in argument lack consistency and lead to inconsistency in the logical coherence of the debate.
3. When there are points that are logically unnatural or difficult to understand:
   - Example: Team A argues for the importance of maximizing a company's profits and provides an example of increasing customer satisfaction as evidence. However, the logical explanation of how these two points are connected is lacking, making it difficult to understand. Consequently, the debate's coherence and logical consistency are compromised.


Please evaluate based on the given debate topic and conversation, according to the rules.

## Input Data


  TOPIC: {topic},
  Debate: {debate}


Below are the criteria for deducting points based on coherence/logic.

## Output Format

{format_instructions}
"""


rebut_template = """You are the judge evaluating the effectiveness of rebuttals when you receive the debate topic and the content of the discussion. Please score it out of 100 points and submit your assessment.

## Rule
Effectiveness of Rebuttals:
1. **Ignoring or Misunderstanding the Opposing Team's Argument when Rebutting:**
   - This occurs when a team fails to adequately address or comprehend the opposing team's argument before attempting to rebut it. For instance, if Team A dismisses or misinterprets Team B's argument without fully understanding its implications, the rebuttal may lack effectiveness.

2. **Rebutting with Weak or Unsupported Evidence:**
   - This happens when a team offers weak or unsubstantiated evidence to counter the opposing argument. For example, if Team B attempts to rebut Team A's argument using faulty logic or outdated data, the rebuttal may not effectively challenge the validity of Team A's position.

3. **Lack of Effective Rebuttal against the Opposing Team's Argument:**
   - This occurs when a team fails to provide a compelling rebuttal against the opposing team's argument. If Team A cannot offer a well-reasoned counterargument to Team B's points, it may weaken their overall position in the debate.

Below are specific examples that could result in deductions. Please consider these when evaluating.

1. **Ignoring or Misunderstanding the Opposing Team's Argument when Rebutting:**
   - Example: Team A failed to properly understand the argument presented by Team B and rebutted it. When Team B emphasized the importance of social responsibility, Team A completely ignored it, focusing instead on gaining economic benefits. As a result, Team A's rebuttal was relatively ineffective.

2. **Rebutting with Weak or Unsupported Evidence:**
   - Example: When rebutting Team A's argument, Team B failed to provide proper evidence or examples and simply denied the argument. Such a rebuttal not only lacks effectiveness but also undermines the credibility of the debate by relying on unsubstantiated claims.

3. **Lack of Effective Rebuttal against the Opposing Team's Argument:**
   - Example: When Team A rebutted Team B's argument, they failed to present proper evidence or logical reasoning and merely offered opposing opinions. This resulted in a lack of effective rebuttal against Team B's argument, leading to a decrease in the logical strength of the debate.


Please evaluate based on the given debate topic and conversation, according to the rules.

## Input Data


  TOPIC: {topic},
  Debate: {debate}


Below are the criteria for deducting points based on coherence/logic.

## Output Format

{format_instructions}
"""

persuasiveness_template = """You are the judge evaluating the persuasiveness when you receive the debate topic and the content of the discussion. Please score it out of 100 points and submit your assessment.

## Rule

**Persuasiveness**
- Assertions made without evidence or factual information to support them.
- Lack of clarity in the argument or insufficient persuasiveness towards the intended audience.
- Excessive use of emotional appeals or exaggeration for the purpose of persuasion.


Below are specific examples that could result in deductions. Please consider these when evaluating.

1. **Assertions without Evidence:**
   - Example: Team A argued for the importance of maximizing company profits without providing specific economic theories or case studies to support their claim. This gave the impression that their argument was merely speculative, undermining the credibility of the debate.

2. **Lack of Clarity or Persuasiveness towards the Target Audience:**
   - Example: Despite Team B's emphasis on the company's social responsibility, their argument lacked clarity and failed to persuade the intended audience effectively. The absence of specific examples or reasons left listeners questioning why they should accept their argument.

3. **Excessive Emotional Appeal or Exaggeration:**
   - Example: Team A emphasized the importance of maximizing company profits using emotional language and exaggerated expressions. However, the excessive use of emotional elements overshadowed factual persuasion, potentially compromising the objectivity of the debate.


Please evaluate based on the given debate topic and conversation, according to the rules.

## Input Data


  TOPIC: {topic},
  Debate: {debate}


Below are the criteria for deducting points based on coherence/logic.

## Output Format

{format_instructions}
"""


info_template = """You are the judge evaluating the accuracy of information when you receive the debate topic and the content of the discussion. Please score it out of 100 points and submit your assessment.

## Rule

Accuracy of Information:
- Using inaccurate or biased information
- Insufficiently verified information or cases
- Inability to distinguish fact from misconception or presentation of false informationL


Below are specific examples that could result in deductions. Please consider these when evaluating.

1. **Using inaccurate or biased information:**
   - Example: When Team A argued for the importanLce of maximizing corporate profits, they cited specific research or data, but the information they referenced was biased or inaccurate. This suggests that their argument lacks factual basis, undermining the credibility of the debate.

2. **Insufficiently verified information or cases:**
   - Example: When Team B emphasized a company's social responsibility, they cited specific cases or studies, but these references were not adequately verified. This indicates that their argument lacks sufficient reliance on credible information, undermining the credibility of the debate.

3. **Inability to distinguish fact from misconception or presentation of false information:**
   - Example: When Team A highlighted the importance of maximizing corporate profits, they failed to clearly distinguish between fact and misconception, presenting false information. For instance, they claimed that maximizing profits automatically fulfills social responsibility, which is untrue and could lead to misconceptions.

Please evaluate based on the given debate topic and conversation, according to the rules.

## Input Data


  TOPIC: {topic},
  Debate: {debate}


Below are the criteria for deducting points based on coherence/logic.

## Output Format

{format_instructions}
"""


coherence_prompt = PromptTemplate(
    template=coherence_template,
    input_variables=[
        "topic",
        "debate",
    ],
    partial_variables={"format_instructions": MY_PARSER.get_format_instructions()},
)
rebut_prompt = PromptTemplate(
    template=rebut_template,
    input_variables=["topic", "debate"],
    partial_variables={"format_instructions": MY_PARSER.get_format_instructions()},
)
persuasiveness_prompt = PromptTemplate(
    template=persuasiveness_template,
    input_variables=["topic", "debate"],
    partial_variables={"format_instructions": MY_PARSER.get_format_instructions()},
)
info_prompt = PromptTemplate(
    template=info_template,
    input_variables=["topic", "debate"],
    partial_variables={"format_instructions": MY_PARSER.get_format_instructions()},
)


class PromptTag(StrEnum):
    COHERENCE = "coherence"
    REBUT = "rebut"
    PERSUASIVENESS = "persuasiveness"
    INFO = "info"


@dataclass
class PromptDto:
    prompt: PromptTemplate
    tag: PromptTag


debate_prompts = [
    PromptDto(prompt=coherence_prompt, tag=PromptTag.COHERENCE),
    PromptDto(prompt=rebut_prompt, tag=PromptTag.REBUT),
    PromptDto(prompt=persuasiveness_prompt, tag=PromptTag.PERSUASIVENESS),
    PromptDto(prompt=info_prompt, tag=PromptTag.INFO),
]
