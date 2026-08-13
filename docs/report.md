# Research Question
Which policyholder and vehicle characteristics best predict whether a customer will file a claim?

## Lift Chart

The model separates the portfolio into risk deciles whose actual claim frequencies range from 0.061 (safest decile) to 
0.195 (riskiest decile) — a 3.19x difference, measured on data the model never saw during training. Frequency increases 
nearly monotonically from decile 0 to decile 9, with the sharpest jump in the top decile alone (from 0.124 in decile 8 
to 0.195 in decile 9), suggesting the model is particularly effective at isolating the highest-risk tail of the 
portfolio rather than finely separating the middle.

# Business Interpretation

**What does this result mean?**
Drivers with a worse claims history (higher BonusMalus), and to a lesser extent younger drivers and older vehicles, 
are more likely to file a claim. The model can sort customers into risk groups where the riskiest group claims about 3 
times as often as the safest group.

**Why should an insurance company care?**
If everyone pays the same price, risky customers are getting a deal and safe customers are overpaying. Over time, the 
safe customers leave for a competitor who prices them fairly, and the company is left mostly with risky customers it's 
underpricing, which can put it out of business.

**How could this affect underwriting(offer cover at all, and on what terms)?** 
The riskiest customers identified by the model could be sent for manual review, offered a higher deductible, or declined
coverage, instead of being automatically approved at a standard rate.

**How could this affect pricing?**
Customers with a worse claims history (higher BonusMalus) or other risk factors identified by the model could be charged
a higher premium than customers who are lower-risk, instead of everyone paying the same amount.

**How could this improve profitability?**
By charging safer customers less and riskier customers more, the company keeps its safe customers while properly 
covering the cost of its risky ones, making the company more profitable.

## Conclusion

BonusMalus, driver age, and vehicle age were the most reliable predictors of claim frequency. The model sorted drivers 
into risk groups where the riskiest group claims about 3.19 times as often as the safest group. BonusMalus in 
particular had a large effect, moving from a low to a high BonusMalus score could multiply a driver's expected claim 
frequency by close to 10 times. If an insurer charged every customer the same flat price, the riskiest customers would 
be underpriced and the safest customers would be overpriced, risking the company to lose its safe customers to 
competitors who price more accurately.
