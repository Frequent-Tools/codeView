
These are questions extracted from here: https://codefinity.com/blog/Top-50-C-plus-plus-Interview-Questions-and-Answers
## Beginner Questions
#### What are the differences between C and C++?
C is primarily a `procedural` language, focusing on function-based programming. In contrast, C++ supports both `procedural` and `object-oriented` programming, allowing for the use of classes and objects. Additionally, C++ includes features which are not present in C.

#### What are the primitive data types in C++?
Primitive data types are built-in or predefined data types and can be used directly by the user to declare variables.

![data_types.png](figures/Top%2050%20C%2B%2B%20Interview%20Questions%20and%20Answers/data_types.png)

#### What is the preprocessor?
The preprocessor in C++ is a component of the compiler that performs preliminary processing of the source code before its actual compilation. It executes various tasks such as defining symbolic constants using #define directives, excluding parts of code using #ifdef, #endif directives, and more. 

#### How does the #include directive work in C++?
The #include directive in C++ is used to include the contents of another file in the current source file. When encountered, the preprocessor replaces the #include line with the contents of the specified file before compilation begins. There are two syntaxes: **<headerfile> for standard library headers and "headerfile" for user-defined headers.** For example, #include "headerfile.h" brings declarations from headerfile.h into the current file.

#### How to protect a header from being included multiple times?
```cpp
#ifndef HEADER_NAME_H
#define HEADER_NAME_H
// (header file content goes here)
#endif
```
As a simpler alternative, many compilers also support:
```cpp
#pragma once
// (header file content here)
```
This achieves the same goal, but is not part of the official C++ standard (although most compilers support it).

#### What is a namespace in C++ and how is it used?
In C++, a namespace is a way to organize and encapsulate code to prevent naming conflicts. It's declared using the namespace keyword, and entities within it are accessed using the scope resolution operator ::.
```cpp
namespace myNamespace {
    // Code goes here
}

myNamespace::myFunction();
```
Namespaces improve code modularity and readability, especially in larger projects or when integrating multiple libraries.

#### How does const affect a variable?
const affects a variable by making it immutable after assigning a value. When a variable is declared as const, it cannot be changed later in the program. Attempting to modify the value of a const variable will result in a compilation error. Using const also helps optimize code and ensures safety by guaranteeing that the variable's value remains unchanged during program execution.

#### How does static affect global/local variables?
In C++, static affects global variables by limiting their visibility to the current source file. For local variables, static changes their storage duration.<br>
**Need more details on this.**

#### What is a pointer variable in C++?
A pointer variable in C++ is a special type of variable that stores the memory address of another variable. Instead of storing the actual value, a pointer variable holds the location (address) in memory where the value is stored. This allows indirect access to the value stored in memory. Pointer variables are declared using an asterisk (*) before the variable name, followed by the data type of the value it points to.
```cpp
int* ptr; // Declaration of a pointer variable that points to an integer
```
To access the value stored at the memory address pointed to by a pointer variable, you dereference the pointer using the asterisk (*) operator. For example:
```cpp
int x = 10;
int* ptr = &x; // Assigning the address of x to ptr
int value = *ptr; // Dereferencing ptr to get the value stored at the address 
```
#### What is the difference between passing parameters by reference and by value in C++?
Passing parameters by value involves making a copy of the actual parameter's value and passing it to the function. This means any modifications made to the parameter within the function do not affect the original variable outside the function.
```cpp
void increment_val(int x) {
    x++;
}

void increment_ptr(int* x) {
    (*x)++;
}

int main() {
    int num = 5;
    // num remains 5 because the function operates on a copy of num
    increment_val(num);
    // num becomes 6 because the function operates on a reference of num
    increment_ptr(&num);
}
```
## Intermediate Questions
#### What is OOP (Object-Oriented Programming)?
OOP (Object-Oriented Programming) is a programming paradigm based on objects and classes. It bundles data and functionality into objects, which interact. Key principles include encapsulation, inheritance, polymorphism, and abstraction, providing a structured approach to software development.

![OOP.png](figures/Top%2050%20C%2B%2B%20Interview%20Questions%20and%20Answers/OOP.png)

#### What is a constructor?
A constructor in C++ is a special member function of a class that is automatically called when an object of that class is created. Its purpose is to initialize the object's data members and set up the object's state. Constructors have the same name as the class and do not have a return type, not even void. They can be overloaded, allowing multiple constructors with different parameter lists to initialize objects in different ways.<br>
Constructors are commonly used to perform tasks such as memory allocation, initialization of member variables, and setting default values.

#### What are the types of constructors in C++?
`Default constructor` is used to create objects when no specific initial values are provided. `Parameterized onstructor` allows you to create objects with given values right from the start. `Copy constructor` helps in making a new object that's an exact copy of another object from the same class. `Copy assignment operator` It's used to set one object's values to another through assignment. `Move constructor` added in C++11, this constructor optimizes resource transfer, making things run faster and more efficiently.

#### What does the inline keyword mean?
The inline keyword in C++ is a specifier used to suggest that a function should be expanded inline by the compiler, rather than being called as a separate function. This can potentially improve performance by avoiding the overhead of a function call. However, it's just a suggestion to the compiler, and the compiler may choose not to inline the function. Typically, small and frequently called functions are good candidates for inlining.

#### What is a delegating constructor?
A delegating constructor in C++ is a constructor within a class that calls another constructor of the same class to perform part of its initialization. Essentially, it delegates some or all of its initialization responsibilities to another constructor within the same class. This allows for code reuse and simplifies the initialization process, especially when multiple constructors share common initialization logic. Delegating constructors were introduced in C++ 11.

#### What is an initializer list?
An initializer list in C++ is a way to initialize the data members of an object or invoke the constructors of its base classes in the constructor initialization list.
```cpp
// Constructor without initializer list
MyClass(int _value, const std::vector<int>& _data) 
{
    value = _value;
    data = _data;
}
```
It consists of a comma-separated list of expressions enclosed in braces { } and is placed after the colon : following the constructor's parameter list. Using an initializer list allows for efficient initialization of data members and base classes, especially for non-default constructible types or const members.
```cpp
// Constructor with initializer list
MyClass(int value, const std::vector<int>& data) 
    : value(value), data(data) {}
```
#### What is the this keyword?
The `this` keyword in C++ is a pointer that points to the current object. It is automatically available within member functions of a class and allows access to the object's own data members and member functions. When a member function is called on an object, this points to that object's memory location, enabling the function to operate on the object's data. The this pointer is useful in situations where member function parameters have the same names as data members, allowing for disambiguation. Additionally, it's used to enable method chaining and to pass the object itself as an argument to other functions.

#### What is STL?
The Standard Template Library (STL) is a collection of generic algorithms and data structures provided as a part of the C++ Standard Library. It includes containers (such as vectors, lists, and maps), algorithms (such as sorting and searching), and iterators (for iterating over elements in containers). STL components are highly efficient, flexible, and reusable, making them a fundamental part of C++ programming. The Standard Template Library (STL) components are typically found in several header files provided by the C++ Standard Library:
```cpp
#include <vector>
#include <list>
#include <deque>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <utility>
```

#### What are the types of polymorphism in C++?
Compile-time Polymorphism resolved during compile-time, includes function overloading and operator overloading. And run-time polymorphism which resolved during run-time, includes virtual functions and function overriding.

#### What is the difference between overload and override?
Overload refers to defining multiple functions or operators with the same name but different parameters within the same scope (e.g., function overloading, operator overloading) while override refers to providing a specific implementation of a function in a derived class that is already defined in its base class, using the override keyword. This is essential for achieving run-time polymorphism through virtual functions.

#### What is an abstract class?
An abstract class in C++ is a class that cannot be instantiated directly and is designed to serve as a base for other classes. It may contain one or more pure virtual functions, which are declared with the virtual keyword and assigned a 0 or = 0 as their body.
```cpp
// Abstract class
class Shape {
public:
    // Pure virtual function
    virtual void draw() const = 0;
};
```
Abstract classes are intended to define an interface or a common set of functionalities that derived classes must implement. They cannot be instantiated on their own but can be used as a base class for creating concrete classes. This concept helps achieve abstraction and polymorphism in object-oriented programming.

#### What is virtual function in C++?
A virtual function in C++ is a member function of a class that is declared with the virtual keyword and can be overridden in derived classes. When a virtual function is called through a base class pointer or reference, the actual implementation of the function is determined at runtime based on the type of the object pointed to or referenced.<br>
Virtual function can be a pure virtual function in C++ if it declared in a base class without providing an implementation, using the virtual keyword followed by = 0 in its declaration. It serves as a placeholder for functionality that **must be implemented by derived classes**, ensuring polymorphic behavior.

#### What types of conversions are present in C++?
C++ supports several types of conversions that allow for the manipulation of data types during program execution. These conversions are crucial for facilitating operations between different types.

![types_of_conversions.png](figures/Top%2050%20C%2B%2B%20Interview%20Questions%20and%20Answers/types_of_conversions.png)

#### What is an exception? How do you throw and catch one?
An exception is an event in a program that disrupts the normal flow of execution. It typically arises from situations like division by zero, file not found, or invalid input. In C++, exceptions are used to handle such anomalies gracefully, allowing the program to continue running or terminate cleanly.<br>
To throw an exception in C++, you use the throw keyword followed by an exception object. This object could be of any data type, including built-in data types, pointers, or objects of a user-defined class.
```cpp
throw "Division by zero error";  // Throwing a string literal
throw -1;                       // Throwing an integer
```

#### What are try-throw-catch blocks?
Try-throw-catch blocks are constructs in C++ used for exception handling. They allow developers to handle runtime errors gracefully by separating error detection (throwing) from error handling (catching).
* **try block:** This block encloses the code where exceptions may occur. It's followed by one or more catch blocks.
* **throw statement:** This statement is used to raise an exception when an error condition is encountered within the try block.
* **catch block:** These blocks follow the try block and are used to catch and handle exceptions thrown within it. Each catch block specifies the type of exception it can catch.

#### What happens if an exception is not caught?
If an exception is thrown in C++ but not caught, it will propagate up the call stack, seeking a matching catch block that can handle it. If no such catch block is found throughout the entire call stack, the program will terminate abnormally. This termination process also includes calling the destructors of all objects that have been fully constructed in the scope between the throw point and the point where the exception exits the program. Additionally, before the program terminates, the C++ runtime system will typically call a special function named std::terminate(). This function, by default, stops the program by calling std::abort(). Therefore, uncaught exceptions lead to the abrupt termination of the program, which can result in a loss of data or other undesirable outcomes. It is generally a good practice to catch and properly handle exceptions to maintain robustness and prevent unexpected behaviors in software.

#### What is the mutable keyword and when should it be used?
The mutable keyword in C++ allows a member of a const object to be modified. It is useful in scenarios where a member variable needs to be changed without affecting the external state of the object, such as when implementing caching mechanisms. A mutable member can be modified even inside a const member function, helping maintain const-correctness of the method while allowing changes to the internal state of the object. For example, it allows updating a cache or a logging counter in an otherwise constant context.

#### What is the friend keyword and when should it be used in C++?
The friend keyword in C++ is used to grant access to private and protected members of a class to other classes or functions. When a class or function is declared as a friend of a class, it has full access to the private and protected members of that class.<br>
However, the friend keyword should be used cautiously as it breaks encapsulation and can make the code less understandable and harder to maintain. Its usage should be limited only to cases where it's truly necessary to achieve the objective.

#### What is a lambda and an anonymous function in C++?
In C++, a lambda expression, also known as an anonymous function, is a convenient way of defining an inline function that can be used for short snippets of code that are not going to be reused elsewhere and therefore do not need a named function. Introduced in C++11, lambdas are widely used for functional programming styles, especially when using standard library functions that accept function objects, such as those in <algorithm> and <functional> modules.
```cpp
// Lambda to print each element multiplied by a captured value
std::for_each(vec.begin(), vec.end(), [multiplier](int n) {
    std::cout << n * multiplier << std::endl;
});
```
#### What is a functor in C++?
In C++, a functor, also known as a function object, is any object that can be used as if it were a function. This is achieved by overloading the operator() of a class.
```cpp
class MultiplyBy {
private:
    int factor; // Member to store the multiplication factor
public:
    MultiplyBy(int x) : factor(x) {} // Constructor to initialize the factor

    // Overloaded operator() allows the object to act like a function
    int operator()(int other) const {
        return factor * other;
    }
};
```
Functors can maintain state and have properties, which distinguishes them from ordinary functions and allows them to perform tasks that require maintaining internal state across invocations or holding configuration settings.

## Advanced Questions

#### How templates work in C++?
In C++, templates provide a way to create generic classes and functions that can work with any data type. They allow you to write code once and use it with different data types without having to rewrite the code for each type.
* **Template Declaration:** To create a template, you use the template keyword followed by a list of template parameters enclosed in angle brackets (<>). These parameters can represent types (typename or class) or non-type parameters (like integers).
* **Template Definition:** You define your class or function as you normally would, but instead of specifying a concrete type, you use the template parameter.
* **Template Instantiation:** When you use the template with a specific data type, the compiler generates the necessary code by replacing the template parameters with the actual types or values.
* **Code Generation:** The compiler generates separate code for each instantiation of the template, tailored to the specific data types or values used.
```cpp
#include <iostream>
// Template function declaration
template <typename T>
T add(T a, T b) {
    return a + b;
}
int main() {
    // Template function instantiation with int
    std::cout << add(5, 3) << std::endl;  // Output: 8
    // Template function instantiation with double
    std::cout << add(3.5, 2.1) << std::endl;  // Output: 5.6
    return 0;
} 
```

#### What is lvalue and rvalue?
In C++, an lvalue refers to an expression that represents an object that occupies some identifiable location in memory (i.e., it has a name). On the other hand, an rvalue refers to an expression that represents a value rather than a memory location, typically appearing on the right-hand side of an assignment expression.
* **Lvalue (Locator value):** Represents an object that has a memory location and can be assigned a value. Examples include variables, array elements, and dereferenced pointers.
* **Rvalue (Right-hand side value):** Represents a value rather than a memory location. It's typically temporary and cannot be assigned to. Examples include numeric literals, function return values, and the result of arithmetic operations.

Understanding the distinction between lvalues and rvalues is important in contexts like assignment, function calls, and overload resolution. It helps determine whether an expression can be assigned to or modified.

#### When can std::vector use std::move?
`std::vector` can use `std::move` when you want to efficiently transfer ownership of its elements to another vector or to another part of your program.
* **Move Semantics:** When you want to transfer the contents of one vector to another efficiently without deep copying. For example:
```cpp
std::vector<int> source = {1, 2, 3};
std::vector<int> destination = std::move(source);
```
* **Returning from Functions:** When a function returns a vector, you can use std::move to transfer ownership of the vector contents efficiently. For example:
```cpp
std::vector<int> createVector() {
    std::vector<int> vec = {4, 5, 6};
    return std::move(vec);
}
```

#### What is metaprogramming?
Metaprogramming is a programming technique where a program writes or manipulates other programs (or itself) as its data. It involves writing code that generates code dynamically during compilation or runtime. Metaprogramming is commonly used in languages like C++ and Lisp, where it enables tasks such as template instantiation, code generation, and domain-specific language creation.

#### What is SFINAE in simple words?
SFINAE stands for "Substitution Failure Is Not An Error." In simple words, SFINAE is a principle in C++ template metaprogramming where if a substitution during template instantiation fails (often due to a type deduction failure), it's not considered a compilation error. Instead, the compiler tries alternative templates or overloads, allowing the program to continue compilation without raising an error.

#### How do I work with build systems like Make and CMake?
To work with build systems like Make and CMake, you first need to understand your project's structure. Then, you write scripts (Makefile for Make, CMakeLists.txt for CMake) to describe how to compile and link your project. After configuring the build environment, you run the build process using commands like `make` (for Make) or `cmake` (for CMake). Pay attention to any errors or warnings during compilation, and make sure your built executable behaves as expected. Additionally, manage project dependencies properly to ensure a smooth build process.

#### What are the characteristics of the std::set, std::map, std::unordered_map, and std::hash containers?
These containers provide different trade-offs in terms of performance and ordering guarantees, allowing developers to choose the most appropriate one based on their specific requirements.

![characteristics_of_containers.png](figures/Top%2050%20C%2B%2B%20Interview%20Questions%20and%20Answers/characteristics_of_containers.png)

#### What is Return Value Optimization (RVO)?
Return Value Optimization (RVO) is a compiler optimization technique in C++ that eliminates unnecessary copying of objects when returning them from a function. Instead of creating a temporary copy of the object and returning it, the compiler constructs the object directly in the memory location of the caller's destination object. This optimization reduces the overhead associated with copying large objects, improving performance and reducing memory usage.

#### What is template specialization?
Template specialization in C++ allows providing custom implementations for templates for specific types or sets of types. It enables tailoring the behavior of a template for particular cases where the default implementation may not be suitable. There are two types: explicit specialization, where you override the default behavior for specific types, and partial specialization, where you specialize the template for a subset of possible template arguments.
```cpp
template<typename T>
struct MyTemplate {
    // Default implementation
};

template<>
struct MyTemplate<int> {
    // Specialized implementation for int
};
```

## FAQs
**Q:** What is the most common type of question asked in C++ interviews?<br>
**A:** Most C++ interviews start with basic questions about syntax, data types, and control structures. They often move into more complex topics like memory management, object-oriented programming principles, and algorithmic challenges.

**Q:** How should I prepare for C++ interview questions?<br>
**A:** Preparation should include a solid understanding of C++ fundamentals, practice with coding problems, and familiarity with common libraries like STL. Reviewing key concepts such as pointers, class inheritance, and polymorphism is also crucial.

**Q:** Are there any specific C++ features I should focus on understanding deeply?<br>
**A:** Yes, mastering topics such as templates, exceptions, and the Standard Template Library (STL) can be particularly beneficial. Understanding modern C++ features introduced in C++11 and later, like smart pointers, lambda expressions, and concurrency features, is also highly recommended.

**Q:** What kind of programming tasks might I expect during a C++ interview?<br>
**A:** You might be asked to solve algorithmic problems using C++, write classes or functions to implement specific behaviors, or refactor existing C++ code to improve its performance or readability. Understanding design patterns and being able to apply them in C++ is also a plus.

**Q:** How important are data structures and algorithms in a C++ interview?<br>
**A:** A strong grasp of data structures (like arrays, lists, stacks, queues, trees, and graphs) and algorithms (such as sorting, searching, recursion, and dynamic programming) is essential, as many technical interview questions revolve around efficiently solving problems using these concepts.
