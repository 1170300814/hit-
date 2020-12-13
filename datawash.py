
def datawash_artist_correct_format(filename):
    loadstring=[]
    with open(filename,"r",encoding="utf-8") as fr:
        for i in fr.readlines():
            if "\t" in i:
                if len(i.split("\t"))==3:
                    loadstring.append(i)

    with open(filename, "w",encoding="utf-8") as fw:
        fw.writelines(loadstring)


datawash_artist_correct_format("user_artist_data.txt")