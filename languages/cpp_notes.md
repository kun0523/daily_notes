- `const`  `constexpr`
  - `const`  运行时常量
  - `constexpr`  编译时常量，即编译完成后就变成常量；
- windows 下 输入流 终止符： `ctrl+z`

# 基础

- 在任何地方都使用**大括号初始化**方法，大括号初始化方法几乎在任何地方都能正常工作，而且它引起的意外也最少；所以被称为“**统一初始化**”


## 变量

- **自动存储期**：对象在声明时在内存中开辟空间，在代码块结束时对象自动销毁，不需要程序员管理资源回收（局部变量就是自动变量）
- **静态存储期**：
  - 静态对象使用 `static` 或 `extern` 关键字来声明
  - `extern` 修饰：在一个cpp文件（编译单元）中声明并定义`extern int MyAge=100;`  在另一个cpp文件（另一个编译单元）中可以访问该变量`extern int MyAge;`
  - 全局静态变量：
    - 在**声明函数的同一范围内**声明静态变量（这一范围即**全局作用域/命名空间作用域**）
    - 全局作用域的静态对象具有静态存储期，**在程序启动时分配，在程序停止时释放**
  - 局部静态变量：
    - **仅局部有效**，不能从函数外部调用，局部静态变量是**在函数作用域内声明的**，就像自动变量一样；
    - 但生命周期是**从包含它的函数第一次调用开始，直到程序退出时结束**；
    - ```cpp
      int print_number(int num) {
        static size_t counter{0};
        ++counter;

        printf("I got a number: %d", num);
        return counter;
      }
      ```

  - 静态成员：
    - 类的成员，与类的任何实例都不关联；
    - 静态成员**本质上类似全局作用域中声明的静态变量和函数**，但必须**使用类名+作用域解析运算符`::`来引用**它们
    -** 必须在全局作用域初始化静态成员，不能在类定义中初始化静态成员**；
    - 静态成员只有一个实例，该类的所有实例都共享同一个静态成员；
    - ```cpp
      struct Food {
        char name[256];
        double price;
        //static int purchase_num{ 0 };  // 静态成员 声明时不能初始化
        static int purchase_num;
      };

      int Food::purchase_num{0};  // 静态成员必须在全局作用域初始化
      ```

- 线程局部存储期
  - 通过为每个线程提供单独的变量副本，来避免多个线程同时对同一个全局变量进行修改；
  - `thread_local` 关键字来修饰
  - 怎么做最后统一的？？

- 动态存储期
  - 手动控制动态对象生命周期何时开始何时结束；
  - 使用`new`表达式，创建给定类型的对象，返回指向新对象的指针；
  - 使用`delete`表达式释放动态对象，`delete`表达式由delete关键字和指向动态对象的指针组成，delete表达式总是返回void;
  - `delete[] my_array;`  使用 delete 释放动态数组
  
几种存储类型的事例代码：
```cpp
struct Tracer {
	Tracer(const char* name_) :name{ name_ } {
		printf("%s Constructed.\n", name);
	}

	~Tracer() {
		printf("%s Destructed.\n", name);
	}

private:
	const char* const name;  // why??  没有两个const就会报错。。。
};


static Tracer t1{"Static Tracer"};  // 静态存储期
static thread_local Tracer t2{"Thread-local Tracer"};  // 线程局部存储期

void main() {
	printf("Start Main Func\n");

	Tracer t3{ "Automatic Tracer" };  // 自动存储期
	printf("t3\n");

	const auto* t4 = new Tracer{ "Dynamic Tracer" };  // 动态存储期
	printf("t4\n");

	delete t4;  // 没有delete表达式，就不会调用析构函数，造成内存泄漏
}
```
输出结果：
```bash
Static Tracer Constructed.
Thread-local Tracer Constructed.
Start Main Func
Automatic Tracer Constructed.
t3
Dynamic Tracer Constructed.
t4
Dynamic Tracer Destructed.
Automatic Tracer Destructed.
Thread-local Tracer Destructed.
Static Tracer Destructed.
```

- 复制语义  copy semantics
  - x 被复制到 y 后，它们是等价且独立的；
  - 对y的修改不会影响x；
  - 从一个内存地址到另一个内存地址的按位复制；
  - 复制构造函数声明方式：`Point(const Point& other);` 注意引用符！！
  - 使用到**复制构造函数**的场景：
    - 通过值传递向函数传参
    - 函数返回一个对象
    - 复制赋值 `Point p2 = p1;`
  - 当对象中的成员是数据量很大的数组时，复制构造函数会耗时很久，此时可以考虑使用`const Point& other` 的方式传参，避免调用复制构造函数；
  - 
  - 复制赋值运算符声明方式：`Point& operator=(const Point& other);` 返回值永远是 `return *this;`;
  - 通常，编译器会为复制构造函数和复制赋值运算符生成默认的实现
    - `Point(const Point& other) = default;`  声明允许产生默认实现
    - `Point& operator=(const Point& other) = default;`  声明允许产生默认实现
  - 如果该类不可以进行复制，可以强制指定不能复制 
    - `Point(const Point& other) = delete;`
    - `Point& operator=(const Point& other) = delete;`

- 移动语义  move semantics
  - 当涉及大量数据时，复制语义可能非常耗时，通常情况下只需要把资源的所有权从一个对象移动到另一个对象；
  - 可以通过 **移动构造函数** 和 **移动赋值运算符** 来指定对象的移动方式；
  - **移动构造函数** 和 **移动赋值运算符** 接收的是右值引用 `const Point&& other`
  - `std::move()` 将变量从左值转变为右值，`#include<utility>`



## for循环

```cpp

// 只能访问，不能修改原序列的值
for(auto i : arr){
    i += 10;
    printf("%d ", i);
}

// 可访问，可修改
for(auto& i : arr){
    i += 10;
    printf("%d ", i);
}
```


```cpp
// 指定格式符
char greet[] = "hello there";
printf("%s\n", greet);
```

- 字符串字面量
```cpp
// 最后一个字符为0，表示字符串结尾；
char alphabeta[27] = {0};   // 如果不给0，会有一串 “烫烫烫烫”
for(int i=0; i<26; ++i){
    alphabeta[i] = 65+i;  // 大写英文字母
}
printf("%s\n", alphabeta);
printf("%d\n", std::size(alphabeta));

for(int i=0; i<26; ++i){
    alphabeta[i] = 97+i;  // 小写英文字母
}
printf("%s\n", alphabeta);
```

## 多态
- 虚方法：
  - 如果想让派生类覆盖基类的方法，可以使用 `virtual` 关键字，如果要求派生类必须实现该方法，在方法后缀加 `=0`(纯虚方法，如果派生类中没有实现该方法，会在编译时报错)；
  - 包含任何纯虚方法的类都不能实例化；
  - 在派生类中重写时，需要在方法声明中加 `override` 关键字
- 接口：
  - 通过从只包含纯虚方法的基类派生来实现接口继承，**在C++中，接口总是纯虚类**（仅包含纯虚方法的类）
  - 通常需要在接口中添加**虚析构函数**，添加纯虚析构函数，可以保证在对象销毁时，调用正确的析构函数；
  - 接口注入方式：构造函数注入 和 属性注入；
    - 构造函数注入：声明一个引用成员，在构造函数初始列表中进行初始化；
    - 属性注入：声明一个指针成员，通过set方法，随时修改该成员的指向；
    - 
### 运行时多态
- 通过对象组合的设计模式，将一个接口类作为类的成员，通过指定接口不同的派生类，在运行时实现多态效果；

### 编译时多态
- 通过模板实现编译时多态

# 自定义类型

## 枚举

```cpp
enum class Week{
  Mon, Tue, Wed, Thu, Fri, Sat, Sun
};

Week day = Week::Mon;
```

## 结构体

```cpp
struct Book{
    char name[256];
    int year;
    int pages;
    bool hardcover;
};

Book b1;
b1.pages = 170;
```


## 类：

- 如果**没有定义构造函数**，可以之间使用 大括号对实例对象进行初始化；
- 如果定义了构造函数，则必然会调用相应的构造函数进行初始化；
- **没有任何大括号或小括号时**，**无参构造函数**被调用；
- 类使用小括号，调无参构造函数，容易被编译器理解为函数声明；

```cpp
class Paper{
public:
    char name[200];
    int year;
    int pages;

    Paper(){
        memcpy(name, "TestPaper01", 20);        
        year = 2024;
        pages = 20;
    }

    Paper(char* name_, int year_, int pages_){
        memcpy(name, name_, 20);        
        year = year_;
        pages = pages_;

    }
};

Paper p1{"TestPaper", 2024, 5};
printf("Paper Name: %s, Year: %d, Pages: %d\n", p1.name, p1.year, p1.pages);

Paper p2{};
printf("Paper Name: %s, Year: %d, Pages: %d\n", p2.name, p2.year, p2.pages);

```

- 一个类的构造函数中，**涉及在堆内存中创建变量**，则需要重载**复制构造函数(CopyConstructor)**和**赋值运算符（=）**和**析构函数（~T）**回收资源
- 将类的构造函数声明为 `explicit` 可以防止进行隐式类型转换；为了避免没有预想到Bug，通常要避免自定义类型的隐式类型转换；
- 

## 静态绑定与动态绑定

- 静态绑定

  - 声明为父类，传值为子类；
  - 调用该对象 父类与子类共有的方法时，默认使用静态绑定；
  - 即调用声明类型的函数定义，即父类的函数定义；

- 动态绑定

  - 声明为父类，传值为子类；

  - 调用 该对象 父类与子类共有的方法时，根据对象内容，对用其对应的方法（在运行时才确定调用哪个函数定义）；

  - 需要在父类与子类，该方法的声明前加 `virtual` 关键字

  - ```cpp
    class Person{
    public:
        Person(std::string name_):name(name_){}
        virtual ~Person(){cout << "destructor Person" << endl;}
        virtual void print(){ cout << "Name: " << name << endl;}
    
    private:
        std::string name;
    
    };
    
    
    // 注意加 public 
    class Student: public Person{
    public:
        Student(std::string name_, int stud_id):Person(name_), studId(stud_id){}
        virtual ~Student(){cout << "destructor Student" << endl;}
        // 加virtual实现动态绑定！！没有virtual 即静态绑定
        virtual void print(){ Person::print(); cout << "ID: " << studId << endl;}
    
    private: 
        int studId;
    };
    
    // ch2
    Person* personArr[10];
    personArr[0] = new Person("xiaowang");
    personArr[1] = new Student("xiaochen", 1);
    
    personArr[0]->print();
    personArr[1]->print();
    
    delete personArr[0];
    delete personArr[1];
    ```

  - 只要类中有定义 `virtual` 方法，就必须定义 `virtual destructor` 

    - 因为如果子类中有使用堆内存，在销毁时就需要对应的回收资源，不能仅靠父类回收资源

  
  
  ## 抽象类
  
  - 不能实例化对象
  - 包含纯虚函数的即为抽象类
  - 类的成员函数，`virtual void func() = 0;` 
  - 
  
  



# 变量

- sizeof	
  - type T， `sizeof(T)`  返回在当前系统下，`T`的大小是几倍的 `char` 的大小，例如 在当前系统下，`char` 是 8bits，`int`是 32bits ，`sizeof(int)` 是 4；
- 声明指针变量
  - `int* x, y, z;`   ==>  `int* x; int y; int z`
    - 同一行中声明多个指针变量时，符号 \* 是与变量名绑定的，所以上式中只有 x 是指针；



## 枚举

```cpp

enum Mood{HAPPY=3, SAD=1, ANXIOUS=4, SLEEPY=2};

int main(){
    Mood myMood = HAPPY;
    Mood yourMood = SAD;
    cout << "HAPPY + SAD = " << myMood+yourMood << endl;
    return 0;
}
```

## std::string

| 方法               | 解释                                                       |
| ------------------ | ---------------------------------------------------------- |
| s.find(p)          | 返回子字符串在s中出现的index， 如果没找到返回 string::npos |
| s.find(p, i)       | 返回索引 i 之后，第一次出现 子串p的索引；                  |
| s.substr(i, m)     | 返回从索引 i 之后的，m个字符组成的子串；                   |
| s.insert(i, p)     | 在指定的索引 i 之前插入子串；                              |
| s.erase(i, m)      | 从索引 i 开始，去除 s 中的 m 个字符                        |
| s.replace(i, m, p) | 使用 字符串p 替换 字符串s 中 m 个字符                      |
| getline(is, s)     | 从 输入流中 读取一行文本；                                 |







# 遍历文件夹

```cpp
std::string image_dir{R"(E:\DataSets\pd_mix\test_pd_images\0416)"};
for(const auto& f : std::filesystem::directory_iterator(image_dir)){
    std::cout << f.path() << std::endl;
}
```



# 自定义谓词排序

```cpp
struct Data{
    int x;
    int y;
    Data(int x_, int y_):x(x_), y(y_){};
};

std::vector<Data> vec;
vec.emplace_back(1,1);
vec.emplace_back(2,1);
vec.emplace_back(1,2);
vec.emplace_back(3,1);

for(const auto& i:vec)
    std::cout << i.x << " " << i.y << std::endl;
std::cout << std::endl;

    std::sort(vec.begin(), vec.end(), [](const Data& a, const Data& b){return a.x<b.x;});

    for(const auto& i:vec)
        std::cout << i.x << " " << i.y << std::endl;
```



# 正则表达式

- `#include <regex>`

- `std::regex pattern("[\u4e00-\u9fa5]");` 这行代码的含义是创建一个正则表达式模式，用于匹配包含任何中文字符的字符串。在程序中，在对应的字符串中，如果包含任何一个中文字符（Unicode 编码范围在 `\u4e00` 到 `\u9fa5` 之间），这个模式会匹配到这个字符

# 容器

## vector

- 初始化方法

  ```cpp
  std::vector<int> vec0;
  std::vector<int> vec1();
  std::vector<int> vec2(10);
  std::vector<int> vec3(10, 2);
  std::vector<int> vec4{ 1,2,3 };
  std::vector<int> vec5 = vec3;
  std::vector<int> vec6(vec3);
  std::vector<int> vec7{ vec3 };
  std::vector<int> vec8(vec3.begin() + 1, vec3.end() - 2);
  ```

- 元素拷贝（不同类型容器之间元素拷贝）

- ```cpp
  std::list<int> lst{ 1,2,3,43,54,65,67,87 };
  std::vector<double> vec1(lst.begin(), lst.end());
  std::vector<int> vi{ 11,22,33,44,55,66 };
  std::vector<double> vec2(vi.begin(), vi.end());
  ```

- 访问元素

  - `vec.at(3)`   会自动做范围检查，如果超出范围会抛异常；
  - `vec[3]`  不做范围检查，当超出范围时，返回默认值；

# 案例

## 拆分文件路径

```cpp
std::string::size_type pos{ 0 }, prev{ 0 };
std::string sep{ "\\" };
std::string file_pth{ R"(D:\share_dir\assy_scf\src\scfAdwithPlc\build\scf_anomaly_detect.exe)" };
// 循环打印文件夹
while ((pos = file_pth.find_first_of(sep, pos)) != std::string::npos) {
    std::string tmp = file_pth.substr(prev, pos-prev);  // 注意substr 第一个参数是起点 第二个参数是*长度*
    std::cout << "Element: " << tmp << std::endl;
    std::cout << "prev: " << prev << " end: " << pos << std::endl;
    prev = ++pos;
}
// 输出最后的文件名
std::string tmp = file_pth.substr(prev);
std::cout << "Element: " << tmp << std::endl;

// 直接找文件名
auto pos = file_pth.rfind('\\');
std::string file_name = file_pth.substr(pos+1);
```



## 监控进程

```cpp
#include <iostream>
#include <windows.h>
#include <tlhelp32.h>

bool IsProcessRunning(const char* processName) {
    // 获取系统中当前正在运行的进程快照
    HANDLE hProcessSnap;
    PROCESSENTRY32 pe32;
    hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (hProcessSnap == INVALID_HANDLE_VALUE) {
        return false;
    }

    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (!Process32First(hProcessSnap, &pe32)) {
        CloseHandle(hProcessSnap);
        return false;
    }

    do {
        if (strcmp(pe32.szExeFile, processName) == 0) {
            CloseHandle(hProcessSnap);
            return true; // 进程在运行
        }
    } while (Process32Next(hProcessSnap, &pe32));

    CloseHandle(hProcessSnap);
    return false; // 进程不在运行
}

void StartProcess(const char* processName) {
    // 强制启动指定的程序
    ShellExecute(NULL, "open", processName, NULL, NULL, SW_SHOWNORMAL);
}

int main() {
    const char* targetProcess = "notepad.exe"; // 替换为你希望监控的程序名称

    if (!IsProcessRunning(targetProcess)) {
        std::cout << targetProcess << "not running." << std::endl;
        StartProcess(targetProcess);
    }
    else{
        std::cout << targetProcess << " is running." << std::endl;
    }

    return 0;
}
```

### 函数解释

- `ShellExecute` 是一个Windows API函数，用于执行外部程序或打开文档文件、URL、文件夹等。它提供了一个简单的方法来启动外部程序或打开文件，让系统自动根据文件类型来选择合适的应用程序进行处理。

- `ShellExecute` 函数的原型如下：

```cpp
HINSTANCE ShellExecute(
  HWND    hwnd,
  LPCTSTR lpOperation,
  LPCTSTR lpFile,
  LPCTSTR lpParameters,
  LPCTSTR lpDirectory,
  INT     nShowCmd
);
```

参数含义如下：

- `hwnd`: 用于接收操作结果和错误信息的窗口句柄。
- `lpOperation`: 指定操作的动词，比如 "open"、"print"、"explore" 等。通常使用 "open" 以启动应用程序或打开文件。
- `lpFile`: 要执行的文件、程序名或URL地址。可以是绝对路径或者相对路径名称。例如 "notepad.exe", "C:\path\to\file.doc", "[http://www.example.com"。](http://www.example.com"./)
- `lpParameters`: 传递给应用程序的参数，如果不需要可以填写为NULL。
- `lpDirectory`: 指定程序的起始工作目录，如果为NULL，则使用当前目录。
- `nShowCmd`: 指定如何显示应用程序的常数，通常可以使用 SW_SHOW 来显示窗口。

下面是一个简单的例子，演示了如何使用`ShellExecute`函数来打开一个网页：

```cpp
#include <Windows.h>

int main() {
    // 打开网页
    ShellExecute(NULL, "open", "https://www.example.com", NULL, NULL, SW_SHOWNORMAL);
    return 0;
}
```

在这个例子中，`ShellExecute` 函数将会调用系统默认的浏览器来打开指定的网页。如果你需要启动一个本地的可执行文件，也可以通过 `ShellExecute` 来实现，方法是将 `lpFile` 参数设置为相应的可执行文件的路径，比如 "C:\path\to\program.exe"。

## 计算函数运行耗时
```cpp
#include <iostream>
#include <chrono>

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    someTask();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t = end - start;
    cout << "cost: " << t.count() << endl;
}
```

# overload override

The terms "overload" and "override" are often used in the context of object-oriented programming, particularly in languages like Java, C++, C#, and others. Here is the difference between the two:

### Overload

1. **Definition**: Overloading refers to the ability to define multiple methods in a class with the same name but with different parameters (different number of parameters or parameter types).
2. **Signature**: Overloaded methods must have unique method signatures (different number or types of parameters).
3. **Compile-Time Polymorphism**: Overloading is an example of compile-time polymorphism (also known as static or early binding) where the compiler determines which method to call based on the method signature.
4. **Return Type**: In overloading, methods with the same name can have different return types, but this alone does not differentiate between them; the parameters are what matter.

Example of method overloading in C#:

```csharp
public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public double Add(double a, double b)
    {
        return a + b;
    }
}
```

### Override

1. **Definition**: Overriding refers to providing a new implementation for a method in a subclass that is already provided in its superclass.
2. **Inheritance**: Override is used to implement runtime polymorphism (also known as dynamic or late binding) to achieve method overriding in an inheritance hierarchy.
3. **Method Signature**: The overridden method in the subclass must have the same method signature (same name, same number and type of parameters) as the method in the superclass.
4. **Annotation**: In languages like C#, the `override` keyword is used to explicitly declare that a method overrides a method in the base class.

Example of method overriding in C#:

```csharp
public class Animal
{
    public virtual void MakeSound()
    {
        Console.WriteLine("Animal sound");
    }
}

public class Dog : Animal
{
    public override void MakeSound()
    {
        Console.WriteLine("Woof!");
    }
}
```

In summary, method overloading is about having multiple methods with the same name but different parameters within the same class, while method overriding is about providing a new implementation for a method in a subclass that is inherited from a superclass.

# 导出DLL

```cpp

struct Point{
    double x;
    double y;
};

extern "C" __declspec(dllexport) void AddPoints(const void* p1, const void* p2, void* res){
    Point* p1_ = (Point*)p1;
    Point* p2_ = (Point*)p2;
    Point* res_ = (Point*)res;

    res_->x = p1_->x + p2_->x;
    res_->y = p1_->y + p2_->y;
    return;
}
```



- `extern "C"`   
  - 以C语言的方式导出函数，不对函数名进行cpp的修饰，便于调用时找到该函数
  - 所以不能使用函数重载特性，两个函数名相同的函数，会导致找不到想要的函数
  - extern  作为外部函数
- `__declspec(dllexport)`
- 
- `__declspec(dllimport)`
- 





# 遇到的BUG

## 处理图像内存泄漏

- 使用opencvsharp读图后，要清理内存，每创建一个Mat都要回收；

  - ```c#
    Mat img = new Mat(imagePth);
    DET_RES[] fpc_det_res = fpc_det_model.doInference(ref img, ref form_msg);
    img.Dispose();
    ```



# Effective C++

## Item 1: View C++ as a federation of languages
- 可以将C++理解为基于四种语句的联合
  - C，保留了C的语句、预处理、内建数据类型、数组、指针等；
  - Object-Oriented C++ 具有类、封装、继承、多态、虚函数（动态绑定）
  - Template C++ 模板编程
  - STL  容器、迭代器、算法、函数对象
  
- 根据不同的业务场景，思路转换到适当的语言风格；

## Item 2: Prefer consts, enums, and inlines to #defines
- 更倾向于编译器，而不是预处理器
- 对于简单的常量，使用 `const` `enum` 而不是 `#define`
- 对于简单的函数，使用 `inline function` 而不是 `#define` 宏函数

## Item 3: Use const whenever possible
- 尽量使用 `const` 修饰；
- 当 `const` 出现在 `*` **左侧**时，代表指针指向的内容为 `const`;
- 当 `const` 出现在 `*` **右侧**时，代表指针本身为 `const`;
- 当 `*` 左右两侧都有 `const` 时，代表 指针 和指针指向的内容 都是 `const` 
    ```cpp
    char greeting[] = "Hello";
    char* p = greeting;  // non-const pointer, non-const data
    const char* p = greeting;  // non-const pointer, const data
    char* const p = greeting; // const pointer, non-const data
    const char* const p = greeting;  // const pointer, const data
    ```
- `const` 出现在变量**类型前后没有差别**
  ```cpp
  void func1(const Widget* pw);
  void func2(Widget const* pw);  // 与上一个声明一致
  ```
- `STL` 中的 `iterator` 就是指针
  ```cpp
  const std::vector<int>::iterator iter = vec.begin();  // 类似 T* const
  *iter = 10;  // OK 可以改变指向的内容；
  ++iter;  // Error  不可以改变指针；

  std::vector<int>::const_iterator cIter = vec.begin(); // 类似 const T*
  *cIter = 10;  // Error  不可以改变内容
  ++cIter;  // OK 可以改变指向
  ```

## Item 4: Make sure that objects are initialized before they are used
- 如果类中成员变量是`const` 或 `reference` 则一定需要初值，不能被赋值，此时需要**只能使用列表初始化**的方式进行初始化

## Item 5: Know what functions C++ silently writes and calls
- 编译器默认为空类写4个方法：
  - `copy` 构造函数
  - `copy assignment` 操作符
  - 析构函数
  - default 构造函数
  ```cpp
  class Empty{};

  // 等同于：
  class Empty{
  public:
      Empty(){...}  // default construct
      Empty(const Empty& rhs){...} // copy construct
      ~Empty(){...}  // deconstruct

      Empty& operator=(const Empty& rhs){...}  // copy assignment operator
  }

  // 方法调用
  Empty e1;  // call default construct
  Empty e2{e1}  // call copy construct
  e2 = e1;  // call copy assignment
  ```



# Effective STL
## Item 1：仔细选择容器

- 标准STL序列容器：vector  string  deque  list
- 标准STL关联容器: set   multiset   map   multimap
- 建议选择方式：
  - `vector` 是一种可以默认使用的序列类型
  - `List`  当很频繁地对序列<font color=red>**中部**</font>进行插入和删除时使用
  - `deque`  当大部分插入和删除发生在序列的<font color=red>**头或尾**</font>时使用

- 考虑算法复杂度
  - 连续内存容器（基于数组的容器）:`vector` `string` `deque`
  - 基于节点的容器：`list`  `slist` `标准关联容器`

## Item 2: 小心对 “容器无关代码” 的幻想
- “容器无关” 是指 后续的代码可以兼容各种容器类型；
- STL建立在泛化之上
  - 数组泛化为容器，参数化了所包含的对象的类型；
  - 函数泛化为算法，参数化了所用的迭代器的类型；
  - 指针泛化为迭代器，参数化了所指向的对象的类型；

## Item 3: 使容器里对象的拷贝操作轻量而正确
- 拷进去，拷出来：向容器中添加一个对象，其实是你指定对象的拷贝，从容器中提取一个对象，取出的是一个对象的拷贝；
- 容器中的元素移动，都是通过拷贝的方式进行的
- 通过类的`拷贝构造函数 Widget(const Widget&)`和它的`拷贝赋值操作 Widget& operator=(const Widget&)`
- **可能导致的问题**
  - 如果容器中的对象，拷贝过程很昂贵，那么容易造成性能瓶颈
  - 如果以基类创建容器，<font color=red>向容器中插入派生类对象时，派生部分会丢失！</font>
  - 较好的解决方法是，创建一个<font color=clane>对象指针的容器</font>
  - 更好的解决方法是，使用智能指针的容器？

## Item 4: 使用`empty`来代替检查`size()`是否为0
- 对于所有的标准容器，`c.empty()`是一个常数时间的操作，对于一些`list` `c.size()` 是线性时间操作；

## Item 5: 尽量使用区间成员函数
- 区间成员函数：使用两个迭代器参数来指定元素的一个区间，来进行某个操作
