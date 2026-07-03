tb = str.maketrans("mrtcw,kae'lndsvyuoighxzbf.-qjp", "qwertyuiopasdfghjkl;zxcvbnm,./")
txt = "This is all dependant on your framework but in expressjs this is how it works (more or less). \n To answer your question, you don't need it - you can simply not issue it as a parameter (although it will still be accessible) if you don't plan on responding (which is weird and quite uncommon)"
txt = "The quick brown fox jumps over the lazy dog"
txt = txt.lower()
print(txt.translate(tb))

