from itertools import combinations

import keyboard
import time

def main():
    name = input("enter file name: ")
    
    with open(name, "a") as f: 
        while True:
            r = keysort()
            if r is None:
                break
            
            k, s = r
            pieces = []
            for i in k:
                if isinstance(i, list):
                    pieces.append(",".join(str(x) for x in i))
                else:
                    pieces.append(str(i))
            
            f.write("|".join(p for p in pieces if p != "\n"))
            f.write("\n")
            f.write(s)
            
def keysort():
    k = {
        "f": 1, "d": 2, "s": 3, "j": 4, "k": 5, "l": 6,
        "space": " "
    }
    
    cs = keypress()
    if cs is None:
        return None
        
    c, s = cs
    d = []

    for i in c:
        b = []
        for j in i:
            if j in k:
                if isinstance(k[j], int):
                    b.append(k[j])
                else:
                    d.append(k[j])
        if b:
            d.append(sorted(b))

    return d, s

def keypress():
    combinations = []
    current_keys = []
    
    print("braille mode")
    
    def on_key_event(event):
        if event.name == "enter":
            return
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "backspace":
                
                combinations.pop()

            current_keys.append(event.name)
        elif event.event_type == keyboard.KEY_UP:
            if current_keys:
                combinations.append(list(current_keys))
                current_keys.clear()
    
    hook = keyboard.hook(on_key_event)
    keyboard.wait("enter")
    keyboard.unhook(hook)
    
    print("typing mode")
    s = input("Enter sentence ")
    
    if s == "555":
        return None
        
    return combinations, s

if __name__ == "__main__":
    main()