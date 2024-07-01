- `const`  `constexpr`
  - `const`  运行时常量
  - `constexpr`  编译时常量，即编译完成后就变成常量；
- windows 下 输入流 终止符： `ctrl+z`

# 类：

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

## Item 3: Use const whenever possible
- 尽量使用 `const` 修饰；

