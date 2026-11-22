| Firm-Level |         |      | Marketing |         |           | Emphasis |      | Data           |     |     |
| ---------- | ------- | ---- | --------- | ------- | --------- | -------- | ---- | -------------- | --- | --- |
| If         | you use | this | dataset   | in your | research, | please   | cite | the following: |     |     |
Damavandi, Hoorsana, Mai, Feng, andAstvansh, Vivek(2025). ANewTechniqueforMeasuring
| a   | Firm’s | Marketing |     | Emphasis. | Marketing |     | Letters. |     |     |     |
| --- | ------ | --------- | --- | --------- | --------- | --- | -------- | --- | --- | --- |
Overview
This dataset provides firm-level scores for various marketing constructs, aggregated by company (GVKEY),
year, and quarter. The scores are derived from textual analysis of corporate earnings call transcripts.
Coverage
337,599 firm-quarter observations from 11,296 U.S. public firms, 2003Q1 to 2025Q4 (Executive file; the
| Analyst |     | file covers | 254,338 | firm-quarters |     | over | the | same period). |     |     |
| ------- | --- | ----------- | ------- | ------------- | --- | ---- | --- | ------------- | --- | --- |
| Version |     | History     |         |               |     |      |     |               |     |     |
• Coverage extended from 2023Q1 to 2025Q4 using the identical published method.
|     | v2  | (August | 2026). |     |     |     |     |     |     |     |
| --- | --- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- |
The 2023Q1 observations in v1 were incomplete due to the timing of the original transcript download
and have been replaced with full-quarter coverage; all observations before 2023Q1 are unchanged from
v1.
• v1 (March 2024). Original release accompanying the paper: 254,522 firm-quarter observations,
|     | 2003Q1 | to  | 2023Q1. |     |     |     |     |     |     |     |
| --- | ------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
Files
1. MarketingEmphasisScores_Executive.csv - Scores based on management presentations and Q&A
|     | responses |     | (average | of both) |     |     |     |     |     |     |
| --- | --------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- |
2. MarketingEmphasisScores_Analyst.csv - Scores based only on analyst questions
Identifiers:
|     | • GVKEY:   |          | Unique   | company | identifier |     |     |     |     |     |
| --- | ---------- | -------- | -------- | ------- | ---------- | --- | --- | --- | --- | --- |
|     | • year:    | Calendar | year     |         |            |     |     |     |     |     |
|     | • quarter: |          | Calendar | quarter |            |     |     |     |     |     |
Other Variables:
• document_length: Total number of words from the transcript segments used to calculate the scores for
|          | the | given | GVKEY,     | year, | and quarter |     |     |     |     |     |
| -------- | --- | ----- | ---------- | ----- | ----------- | --- | --- | --- | --- | --- |
| Measured |     |       | Constructs |       |             |     |     |     |     |     |
1. (8 dimensions): customer_orientation, competitor_orientation,
|     | Marketing |     | Orientation |     | Measures |     |     |     |     |     |
| --- | --------- | --- | ----------- | --- | -------- | --- | --- | --- | --- | --- |
interfunctional_coordination, longterm_focus, profit_focus, intelligence_generation, intelli-
gence_dissemination, responsiveness (Deshpandé and Farley 1998; Kohli and Jaworski 1990; Narver
|     | and | Slater | 1990) |     |     |     |     |     |     |     |
| --- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- |
2. Marketing Capabilities Measures (8 dimensions): info_management, planning, implementation,
pricing_capabilities, product_development, channel_management, communication, selling_capabilities
|     | (Day      | 1994; | Vorhies    | and | Morgan     | 2005) |     |              |                      |                   |
| --- | --------- | ----- | ---------- | --- | ---------- | ----- | --- | ------------ | -------------------- | ----------------- |
|     | 3.        |       |            |     |            |       | (3  | dimensions): | marketing_ecosystem, | end_user, market- |
|     | Marketing |       | Excellence |     | Dimensions |       |     |              |                      |                   |
ing_agility (Homburg, Theel, and Hohenberg 2020; Moorman and Day 2016)
1

| Variable  | Types        |                    |                    |            |
| --------- | ------------ | ------------------ | ------------------ | ---------- |
| Each file | contains two | types of variables | for each marketing | construct: |
1. Raw scores: Original measures (e.g., customer_orientation, competitor_orientation)
2. ZCA-whitened scores: Transformed measures with zca_ prefix (e.g., zca_customer_orientation,
zca_competitor_orientation)
The ZCA-whitened variables are orthogonal transformations that preserve semantic meaning while creating
distinct measures that capture unique aspects of marketing emphasis (Kessy et al. 2016). The Marketing
Orientation measures, Marketing Capabilities measures, and Additional Marketing Dimensions are each
| transformed | as separate | groups. |     |     |
| ----------- | ----------- | ------- | --- | --- |
References
Day, G. S. (1994). The capabilities of market-driven organizations. Journal of Marketing, 58(4), 37-52.
Deshpandé, R., & Farley, J. U. (1998). Measuring market orientation: Generalization and synthesis. Journal
| of Market | Focused Management, | 2(1), 213-32. |     |     |
| --------- | ------------------- | ------------- | --- | --- |
Homburg, C., Theel, M., & Hohenberg, S. (2020). Marketing excellence: Nature, measurement, and investor
| valuations. | Journal of | Marketing, 84(4), | 1-22. |     |
| ----------- | ---------- | ----------------- | ----- | --- |
Kessy, A., Lewin, A., & Strimmer, K. (2016). Optimal Whitening and Decorrelation. arXiv:1512.00809v4.
Kohli, A. K., & Jaworski, B. J. (1990). Market orientation: The construct, research propositions, and
| managerial | implications. | Journal of Marketing, | 54(April), | 1-18. |
| ---------- | ------------- | --------------------- | ---------- | ----- |
Moorman, C., & Day, G. S. (2016). Organizing for marketing excellence. Journal of Marketing, 80(6), 6-35.
Narver, J. C., & Slater, S. F. (1990). The effect of market orientation on business profitability. Journal of
| Marketing, | 54(4), 45-78. |     |     |     |
| ---------- | ------------- | --- | --- | --- |
Vorhies, D. W., & Morgan, N. A. (2005). Capabilities for sustainable competitive advantage. Journal of
| Marketing, | 69(January), | 80-94. |     |     |
| ---------- | ------------ | ------ | --- | --- |
2