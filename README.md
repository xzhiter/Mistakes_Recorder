# 错题整理
程序启动后，需要进行较长时间的初始化。

初始化结束后，您需要在控制台中输入作业扫描件（支持jpg/png/bmp/pdf格式）所在的目录。

一段时间后，程序将显示作业的切题扫描结果，您需要点击您需要整理的错题。如果程序扫描的切题矩形大小不合适，您可以将鼠标指针移到需要修改大小的矩形内，按WSAD按键扩大矩形，或按IKJL缩小矩形。如果有多余的矩形，您可以将鼠标指针移到这些矩形内，然后按下Delete按键。

您可以按键盘上的左右方向键来切换页面。

错题选择结束后，请关闭错题选择窗口（不是控制台窗口），程序会将所选错题的打印件（docx格式）保存到程序所在目录。

# 注意事项
此程序的发行版基于Python 3.9+，不能在Windows 7及之前的系统上运行。

此程序会调用百度的通用文字识别（高精度带位置版）接口，程序运行时请保持网络畅通。接口调用限制为500次/月。
