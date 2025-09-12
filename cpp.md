
#### C/C++ 最常见50道面试题
https://blog.csdn.net/BostonRayAlen/article/details/93041395



#### A pointer to a const object and a const pointer to an object. 
```cpp
    const Animal* pAnimal = &cat;
    cout<<pAnimal->getName()<<" says "<<pAnimal->speak()<<endl;
    pAnimal = &dog;
    cout<<pAnimal->getName()<<" says "<<pAnimal->speak()<<endl;
```
This Code snippet works. The pointer pAnimal itself is not const; it can be changed to point to another object. The object pointed to by pAnimal is considered const through this pointer, meaning you cannot modify the object through pAnimal.

```cpp
    Animal* const pAnimal = &cat;
    cout<<pAnimal->getName()<<" says "<<pAnimal->speak()<<endl;
    pAnimal = &dog;
    cout<<pAnimal->getName()<<" says "<<pAnimal->speak()<<endl;
```
This code snippet doesn't work. The pointer is declared as a const pointer to Animal and couldn't point to a different object after initialization. <br>

If the pointer were declared as const Animal* const pAnimal, then pAnimal would be a constant pointer to a constant Animal, meaning neither the pointer nor the object it points to can change through the pointer.







