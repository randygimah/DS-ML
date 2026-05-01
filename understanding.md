Understanding axes.flatten in day3mod6.ipynb
axes.flatten() reshapes a 2D array into a 1D array. 
When you create plt.subplot(2,3) it returns a 2D array of axes = [
  [ax_row0_col0, ax_row0_col1, ax_row0_col2],   # row 0
  [ax_row1_col0, ax_row1_col1, ax_row1_col2]    # row 1
]
To access each subplot, you'd need 2D array of axes:
axes[0][0]  # top-left
axes[0][1]  # top-middle
axes[0][2]  # top-right
axes[1][0]  # bottom-left
axes[1][1]  # bottom-middle
axes[1][2]  # bottom-right
Looping will be messy = 
for i in range(2):
    for j in range(3):
        sns.regplot(..., ax=axes[i][j])  # nested loops needed
With .flatten, it is converted to 1D 
axes = [ax_row0_col0, ax_row0_col1, ax_row0_col2, ax_row1_col0, ax_row1_col1, ax_row1_col2]
We can then access with 1 index:
axes[0]  # top-left
axes[1]  # top-middle
axes[2]  # top-right
axes[3]  # bottom-left
axes[4]  # bottom-middle
axes[5]  # bottom-right
Looping is simple
for i, col in enumerate(cols):
    sns.regplot(..., ax=axes[i])  # single index, clean!