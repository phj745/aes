system_cot_content="""##content##Your goal is to predict the score an essay received from its text. You should think step by step, reason thoroughly but output ##in a concise way## and then make the conclusion.##content##
"""

system_cot_standard = """##standard##
 Score of 6 (Excellent)
- Clear, consistent mastery with few minor errors
- Insightful point of view and strong critical thinking
- Well-organized, coherent, and focused
- Excellent use of evidence from source texts
- Sophisticated, varied, and precise language

 Score of 5 (Strong)
- Generally strong with occasional minor errors
- Effective point of view and good critical thinking
- Clear organization and logical flow
- Appropriate use of evidence from source texts
- Good language use with some variety

 Score of 4 (Competent)
- Adequate but with noticeable lapses
- Clear point of view and fair critical thinking
- Mostly organized and focused
- Sufficient evidence from source texts
- Acceptable language use with some errors or repetition

 Score of 3 (Developing)
- Inconsistent or limited thinking
- Weak or underdeveloped support from texts
- Limited organization or coherence
- Uneven or basic language with errors
- Vocabulary or sentence variety may be lacking

 Score of 2 (Weak)
- Vague or unclear point of view
- Inadequate or off-topic evidence
- Poor organization and logic
- Frequent language and grammar issues
- Vocabulary is limited or often incorrect

 Score of 1 (Very Weak)
- No clear position or logic
- Little to no relevant support
- Disorganized and hard to follow
- Severe language and grammar errors
- Meaning often unclear
##standard##
"""
system_cot_tips="""##tips##
You should provide some examples in the text when you are reasong if needed.
##tips##
"""
system_cot_format="""
##format##
Reason step by step and place the thought process within the <think></think> tags with steps and output 1-6 ##directly## within the <conclusion></conclusion> tags
<think>
 1.Mastery
 2.Point of view
 3.Structure
 4.Errors
</think>
<conclusion>

</conclusion>
##format
"""
system_cot_gt="""##gt##You have known the answer that the score of this text is ##{score} ##, but you still need to follow the instruction and reason step by step to make the conclusion##gt##"""
system_cot_label=system_cot_content+system_cot_standard+system_cot_tips+system_cot_format+system_cot_gt
system_cot_infer=system_cot_content+system_cot_format
system_cot_infer_tips=system_cot_label.replace(system_cot_gt,"")
system_analyse="""

"""
