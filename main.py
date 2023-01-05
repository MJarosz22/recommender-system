import svd
import item

import pandas as pd

svd_pred = svd.create_svd_predictions()
item_pred = item.create_item_predictions()
print(svd_pred)
print(item_pred)

# Combine the results
combined = []
for i in range (0, len(svd_pred)):
    combined.append((svd_pred[i][0], svd_pred[i][1] * 0.85 + item_pred[i][1] * 0.15))

# Create submission file
submission = pd.DataFrame(combined, columns=['Id', 'Rating'])
submission.to_csv('submission_combined.csv', index=False)