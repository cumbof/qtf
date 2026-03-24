progress = {
"E0":4.8,
"E1":4.0,
"E2":3.1,
"E3":3.0,
"E4":2.8,
"E5":2.7,
"E6":2.6,
"E7":2.5,
"E8":2.4,
"E9":2.24
}

import matplotlib.pyplot as plt

plt.bar(progress.keys(), progress.values())
plt.ylabel("Best RMSD (Å)")
plt.title("Chignolin Folding Progress")
#plt.show()
plt.savefig('chignolin_progress.png')
