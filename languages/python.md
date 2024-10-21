# Python

## Ubuntu使用Python

### 搭建虚拟环境
```bash
sudo apt update
sudo apt install python3 python3-venv
mkdir proj1
cd proj1
python3 -m venv env
source env/bin/activate
pip install packagename
deactivate

rm -rf env  // 删除指定的虚拟环境
```