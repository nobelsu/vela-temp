def readText(path: str, encoding: str = "utf-8") -> str:
    with open(path, mode="r", encoding=encoding) as f:
        return f.read()
    
def writeText(path: str, content: str, encoding: str = "utf-8") -> None:
    with open(path, mode="w", encoding=encoding, newline="\n") as f:
        f.write(content)
