import os
from collections import Counter


Theme = []
with open(os.path.join(os.getcwd(), "plot", "Theme.md"), "r") as f:
    for theme in list(filter(None, " ".join("".join(f.readlines()).split()).split("---")[1:])):
        Theme.append([])
        for color in list(filter(lambda x: x.strip(), theme.split("$"))):
            Theme[-1].append(list(filter(None, color.split("\\")))[0].split("{")[-1][:-1])
    Theme = list(filter(None, Theme))


Sequential_Theme = [
    ["#E4F1F7", "#C5E1EF", "#9EC9E2", "#6CB0D6", "#3C93C2", "#226E9C", "#0D4A70"]
    , ["#E1F2E3", "#CDE5D2", "#9CCEA7", "#6CBA7D", "#40AD5A", "#228B3B", "#06592A"]
    , ["#F9D8E6", "#F2ACCA", "#ED85B0", "#E95694", "#E32977", "#C40F5B", "#8F003B"]
    , ["#B7E6A5", "#7CCBA2", "#46AEA0", "#089099", "#00718B", "#045275", "#003147"]
    , ["#FCE1A4", "#FABF7B", "#F08F6E", "#E05C5C", "#D12959", "#AB1866", "#6E005F"]
    , ["#FFF3B2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"]
]
Projects_Theme = [
    ["#FDFCE8", "#F1F3E5", "#E4E9E2", "#D7DFDF", "#CAD5DB", "#BDCBD8", "#B1C2D5", "#A4B8D2", "#97AECF", "#8AA4CB"]
    , ["#E6F1E9", "#EAF3E8", "#F0F4E6", "#F7F6E6", "#F5F2DF", "#F7E8D5", "#EDD5C5", "#DCBEB0", "#B59790", "#D6C2C0"]
]

Uniform_CMAP = ["viridis", "plasma", "inferno", "magma", "cividis"]
Sequential_CMAP = ["Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"]

Font = {
    "family": "Times New Roman"
    , "size": 14
}


def main():
    print(sorted(Counter([len(theme) for theme in Theme]).items(), key=lambda x: x[0]))


if __name__ == "__main__":
    main()