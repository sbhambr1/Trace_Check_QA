import os
from openai import OpenAI

api_key=os.environ["DEEPSEEK_API"]
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(f"Reasoning: {reasoning_content}")
print("--------------------")
print(f"Response: {content}")

"""
Sample Output:

Reasoning: Okay, so I need to figure out whether 9.11 is greater than 9.8 or if 9.8 is greater than 9.11. Hmm, let's start by looking at both numbers. They both have a 9 in the ones place, so that part is equal. Now, the difference must be in the decimal parts. 

First, I should compare the tenths place. For 9.11, the tenths digit is 1, and for 9.8, the tenths digit is 8. Wait, but hold on, 9.8 can also be written as 9.80 to make the decimal places the same. That might make it easier to compare. So, 9.11 versus 9.80. 

Comparing the tenths place: 1 versus 8. Since 1 is less than 8, that would mean that 9.11 is less than 9.80. But let me double-check that. If I think about money, for example, $9.11 is nine dollars and eleven cents, and $9.80 is nine dollars and eighty cents. Clearly, eighty cents is more than eleven cents, so $9.80 is more. That makes sense. 

Alternatively, maybe I can subtract them to see the difference. Let's do 9.8 minus 9.11. To subtract these, I should line them up:

```
 9.80
-9.11
------
 0.69
```

So, 9.8 minus 9.11 is 0.69, which is a positive number. That means 9.8 is larger by 0.69. 

But wait, another way to look at it is to convert them to fractions. Let's see. 9.11 is the same as 9 + 11/100, and 9.8 is 9 + 8/10. If I convert 8/10 to hundredths, that's 80/100. So 9.8 is 9 + 80/100, which is 9.80. Comparing 80/100 and 11/100, clearly 80 is bigger. So 9.80 is greater than 9.11. 

I think that's solid. But just to make sure, maybe I can think of them as mixed numbers. 9.11 is 9 11/100, and 9.8 is 9 8/10, which simplifies to 9 4/5. If I convert 4/5 to hundredths, multiply numerator and denominator by 20: 4/5 = 80/100. So 9 4/5 is 9 80/100. Comparing 9 80/100 and 9 11/100, again 80 is more than 11, so 9.8 is greater. 

Is there any chance I might have confused the decimal places? For example, if someone reads 9.11 as 9 and 11 tenths, but that's not right. The first decimal place is tenths, the second is hundredths. So 9.11 is 9 + 1/10 + 1/100, while 9.8 is 9 + 8/10. 

Another angle: maybe visualize them on a number line. Between 9.0 and 10.0, 9.11 is just a little past 9.1, whereas 9.8 is much closer to 10.0. So definitely, 9.8 is further to the right, meaning it's larger. 

Wait, just to confirm with another method. Let's multiply both numbers by 100 to eliminate the decimals. 9.11 * 100 = 911, and 9.8 * 100 = 980. Now compare 911 and 980. 980 is clearly larger. So 9.8 is greater. 

Alternatively, if I consider significant digits or something, but I don't think that's relevant here. Both numbers have two digits after the decimal if we write 9.8 as 9.80. So, straightforward comparison. 

I think all these methods consistently show that 9.8 is greater than 9.11. I can't see any error in my reasoning here. Each approach—subtracting, converting to fractions, using money as an analogy, multiplying to eliminate decimals—leads to the same conclusion. So I'm pretty confident that 9.8 is the greater number.
--------------------
Response: To determine which number is greater between **9.11** and **9.8**, follow these steps:

1. **Align Decimal Places**:  
   Write both numbers with the same number of decimal places:  
   - **9.11** remains **9.11**  
   - **9.8** becomes **9.80**

2. **Compare Digit by Digit**:  
   - **Ones place**: Both have **9** (equal).  
   - **Tenths place**: **1** (from 9.11) vs. **8** (from 9.80). Since **1 < 8**, **9.80** is larger here.  

3. **Alternative Verification**:  
   - **Subtraction**: \( 9.80 - 9.11 = 0.69 \) (positive result confirms \( 9.80 > 9.11 \)).  
   - **Fraction Conversion**:  
     - \( 9.11 = 9 + \frac{11}{100} \)  
     - \( 9.8 = 9 + \frac{80}{100} \).  
     Since \( \frac{80}{100} > \frac{11}{100} \), \( 9.8 > 9.11 \).  

**Conclusion**: **9.8** is greater than **9.11**.

\(\boxed{9.8}\)
"""

# # Round 2
# messages.append({'role': 'assistant', 'content': content})
# messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
# response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=messages
# )