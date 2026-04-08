# Error Analysis

        ## Category model test metrics

        - Accuracy: 0.7736
        - Precision (macro): 0.7330
        - Recall (macro): 0.8079
        - F1 (macro): 0.7339
        - Error rate: 22.64%

        ## Priority model test metrics

        - Accuracy: 0.8967
        - Precision (macro): 0.7578
        - Recall (macro): 0.7012
        - F1 (macro): 0.7124
        - Error rate: 10.33%

        ## Top category confusions

        - Actual 'General Query' predicted as 'Billing': 679
- Actual 'General Query' predicted as 'Technical Issue': 538
- Actual 'Technical Issue' predicted as 'General Query': 533
- Actual 'Technical Issue' predicted as 'Billing': 364
- Actual 'Account' predicted as 'General Query': 253

        ## Top priority confusions

        - Actual 'Medium' predicted as 'High': 367
- Actual 'Low' predicted as 'High': 325
- Actual 'High' predicted as 'Medium': 256
- Actual 'Low' predicted as 'Medium': 114
- Actual 'High' predicted as 'Low': 87

        ## Sample category errors

        ```text
                                                                                                                         ticket_text actual_category predicted_category
    access to project user provide attributes explorer attribute explorer matrix content note please attach newly completed          Account      General Query
    user location march user location hello please attached take relevant action proper order location populated ad creates          Account      General Query
I'm having an issue with the {product_purchased}. Please assist.\n\n{product_purchased}\n\n{"name":"New Pest Control Kit, 8    General Query            Billing
    I'm facing a problem with my {product_purchased}. The {product_purchased} is not turning on. It was working fine until y Technical Issue            Billing
  I'm having an issue with the {product_purchased}. Please assist. No sales will be made.\n\nPlease do not order this item u   General Query            Billing
        ```
