with open("datasets/OCA_2.0.pdn", "r") as f:
    a = f.readlines()
a = "".join(a)
# print(a)
b = a.split('\n\n')
# print(b[-1])
c = [game[game.rindex(']')+2:] for game in b[:-1]]
c = [val.replace("\n"," ").split(" ") for val in c]
# l = c[0].replace("\n"," ").split(" ")
# k = " ".join([val for val in l if not val.endswith(".")][:-1])
all_games = "\n".join([" ".join([val for val in l if not val.endswith(".")][:-1]) for l in c])

with open("datasets/real_games.txt", "w") as f:
    f.write(all_games)