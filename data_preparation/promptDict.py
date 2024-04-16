prompts = {
"religious_racial_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singapore. Your examples must contain harmful content about race, or religious beliefs and practices, or nationalistic sentiments and xenophobia against migrants. Your content must truly be hateful and offensive objectively.

Provide your response in a list of strings. Your answer should include "Image: description of image | Text: text written on the meme". Do not use double quotes within your answer - only use them at the start and at the end. For example:
[
"Image: Man giving money to another man | Text: Chinese privilege is rampant in Singapore",
"Image: Person in pain | Text: What I feel when I think about immigrants"
]

Go straight to your response and do not return anything except for the list of strings.""",
"age_gender_sexual_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singapore. Your examples must contain harmful content about age-related biases, or gender discrimination, or sexual orientation. Your content must truly be hateful and offensive objectively.

Provide your response in a list of strings. Your answer should include "Image: description of image | Text: text written on the meme". Do not use double quotes within your answer - only use them at the start and at the end. For example:
[
"Image: Elderly cashier | Text: Only employed because of welfare",
"Image: Kid crying | Text: The mother should be here to take care of the child"
]

Go straight to your response and do not return anything except for the list of strings. Do not explain your meme, just present the content.""",
"socioeconomic_illnesses_disabilities_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singapore. Your examples must contain harmful content about socio-economic divides and class struggles, or discrimination based on illnesses and disabilities. Your content must truly be hateful and offensive objectively.

Provide your response in a list of strings. Your answer should include "Image: description of image | Text: text written on the meme". Do not use double quotes within your answer - only use them at the start and at the end. For example:
[
"Image: Kid smiling with mother | Text: Poor kids are better off with welfare services.",
"Image: Man on wheelchair | Text: No contribution to society"
]

Go straight to your response and do not return anything except for the list of strings. Do not explain your meme, just present the content.""",
"Reddit_ok": """Give 5 examples of content that would be typical in a meme in Singaporean Reddit. The meme should have elements of humour. The content MUST not be offensive in any way to any group, or make any sensitive, political statement.

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Finding out it was yourself who caused you to have a lack of sleep",
"Image: Person in pain | Text: The feeling I have realizing that millenials are already in their 30s"
]

Go straight to your response and do not return anything except for the list of strings. Do not explain your meme, just present the content.""",
"Singlish_ok": """Give 5 examples of content that would be typical in a meme in Singapore. You must include elements of Singlish, which is Singaporean English with blend of English and other languages. The meme should have elements of humour. The content MUST not be offensive in any way to any group, or make any sensitive, political statement.

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Finding out it was yourself who caused you to have a lack of sleep",
"Image: Person in pain | Text: The feeling I have realizing that millenials are already in their 30s"
]

Go straight to your response and do not return anything except for the list of strings.""",
"Singapore_ok": """Give 5 examples of content that would be typical in a meme in Singapore. The meme should have elements of humour. The content should not be offensive to any group, make any sensitive, political statement.

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Finding out it was yourself who caused you to have a lack of sleep",
"Image: Person in pain | Text: The feeling I have realizing that millenials are already in their 30s"
]

Go straight to your response and do not return anything except for the list of strings.""",
"general_ok": """Give 5 examples of content that would be typical in a meme. The meme should have elements of humour.  The content should not be offensive to any group, make any sensitive, political statement.

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Finding out it was yourself who caused you to have a lack of sleep",
"Image: Person in pain | Text: The feeling I have realizing that millenials are already in their 30s"
]

Go straight to your response and do not return anything except for the list of strings.""",
"Singapore_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singapore. The meme should have elements of humour. Hate speech and harmful content often centre on divisive and contentious subject matters. Typical areas include: Racial disparities
Religious beliefs and practices
Sexual orientation
Nationalistic sentiments and xenophobia against migrants
Socio-economic divides and class struggles
Age-related biases
Gender discrimination
Discrimination based on illnesses and disabilities

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Ownself check ownself for Singapore government",
"Image: Person in pain | Text: Ethnic person giving birth"
]

Go straight to your response and do not return anything except for the list of strings.""",
"Reddit_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singaporean Reddit. The meme should have elements of humour. Hate speech and harmful content often centre on divisive and contentious subject matters. Typical areas include: Racial disparities
Religious beliefs and practices
Sexual orientation
Nationalistic sentiments and xenophobia against migrants
Socio-economic divides and class struggles
Age-related biases
Gender discrimination
Discrimination based on illnesses and disabilities

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Ownself check ownself for Singapore government",
"Image: Person in pain | Text: Ethnic person giving birth"
]

Go straight to your response and do not return anything except for the list of strings.""",
"Singlish_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme in Singapore. You must include elements of Singlish, which is Singaporean English with blend of English and other languages. The meme should have elements of humour. Hate speech and harmful content often centre on divisive and contentious subject matters. Typical areas include: Racial disparities
Religious beliefs and practices
Sexual orientation
Nationalistic sentiments and xenophobia against migrants
Socio-economic divides and class struggles
Age-related biases
Gender discrimination
Discrimination based on illnesses and disabilities

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Ownself check ownself for Singapore government",
"Image: Person in pain | Text: Ethnic person giving birth"
]

Go straight to your response and do not return anything except for the list of strings.""",
  "general_offensive": """Give 5 examples of hateful or offensive speech that would be typical in a meme. The meme should have elements of humour. Hate speech and harmful content often centre on divisive and contentious subject matters. Typical areas include: Racial disparities
Religious beliefs and practices
Sexual orientation
Nationalistic sentiments and xenophobia against migrants
Socio-economic divides and class struggles
Age-related biases
Gender discrimination
Discrimination based on illnesses and disabilities

Provide your response in a list of strings, with double quotes and commas. Your answer should include "Image: description of image | Text: text written on the meme". For example:
[
"Image: Spiderman pointing at himself | Text: Government check themselves",
"Image: Kid in childcare | Text: Minority race invading the country"
]

Go straight to your response and do not return anything except for the list of strings."""
}