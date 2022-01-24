
win10上安装opencv详细教程（超详细！！！小白专用！！！）
https://blog.csdn.net/m0_47854694/article/details/115261082

先安装好opencv，再用文本编辑器（如记事本）打开ppyolo_opencv.sln
修改
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "ppyolo_opencv", "ppyolo_opencv.vcxproj",
中的ppyolo_opencv为你喜欢的项目名（可以不改）。

用文本编辑器（如记事本）打开ppyolo_opencv.vcxproj
修改
  <PropertyGroup Label="Globals">
    ...
    <RootNamespace>ppyolo_opencv</RootNamespace>
    ...
  </PropertyGroup>
中的ppyolo_opencv为你喜欢的RootNamespace（可以不改）。

修改
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LinkIncremental>true</LinkIncremental>
    <IncludePath>D:\opencv\build\include;D:\opencv\build\include\opencv2;$(IncludePath)</IncludePath>
    <LibraryPath>D:\opencv\build\x64\vc15\lib;$(LibraryPath)</LibraryPath>
  </PropertyGroup>
中的IncludePath、LibraryPath中涉及到的3个opencv的路径为你安装的opencv的路径。

修改
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      ...
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalDependencies>opencv_world455d.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
中的AdditionalDependencies的opencv_world455d.lib为{你安装的opencv的路径}\build\x64\vc15\lib（比如D:\opencv\build\x64\vc15\lib）目录下opencv_world???d.lib文件的名字。

最后，用vs2019打开ppyolo_opencv.sln


