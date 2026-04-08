# Error Analysis

        ## Category model test metrics

        - Accuracy: 0.8114
        - Precision (macro): 0.7403
        - Recall (macro): 0.7244
        - F1 (macro): 0.7318
        - Error rate: 18.86%

        ## Priority model test metrics

        - Accuracy: 0.9899
        - Precision (macro): 0.9949
        - Recall (macro): 0.7500
        - F1 (macro): 0.8308
        - Error rate: 1.01%

        ## Top category confusions

        - Actual 'General Query' predicted as 'Technical Issue': 20
- Actual 'Technical Issue' predicted as 'General Query': 14
- Actual 'Account' predicted as 'Technical Issue': 7
- Actual 'Technical Issue' predicted as 'Account': 6
- Actual 'Account' predicted as 'General Query': 5

        ## Top priority confusions

        - Actual 'High' predicted as 'Medium': 3

        ## Sample category errors

        ```text
                                                                            ticket_text actual_category predicted_category
                 [TICKET ID] - New Support Ticket received  - Mobile connection   General Query    Technical Issue
           Ajuste FoneBoa tarde      Serviço ja ajustado do Fone.      Obrigado   General Query    Technical Issue
                                   Check[LOCATION]for One Drive ([SERVER]-TEST)         Account    Technical Issue
                                                         email group membership Technical Issue            Account
[TICKET ID] - New Support Ticket received  - Approval to access [NAME]s mailbox Technical Issue      General Query
        ```
