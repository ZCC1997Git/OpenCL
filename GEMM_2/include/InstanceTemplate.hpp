#pragma once
#include <assert.h>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

template <class... INSTANCE>
void InstanceTemplate(std::string Kernelname,
                      std::string& KernelSource,
                      INSTANCE... instance) {
    auto pos = KernelSource.find(Kernelname);
    auto template_end = KernelSource.rfind(">", pos);
    auto template_begin = KernelSource.rfind("<", pos);
    /*get the tempalte paramter*/
    auto All_template_paramter = KernelSource.substr(
        template_begin + 1, template_end - template_begin - 1);

    /*get the seperated paramter*/
    std::vector<std::string> template_paramter;
    size_t pos_template = 0;
    while (pos_template < All_template_paramter.size()) {
        auto start = All_template_paramter.find(",", pos_template);
        if (start == std::string::npos) {
            break;
        }
        template_paramter.push_back(
            All_template_paramter.substr(pos_template, start - pos_template));
        pos_template = start + 1;
    }
    template_paramter.push_back(All_template_paramter.substr(
        pos_template, All_template_paramter.size() - pos_template));

    for (auto& name : template_paramter) {
        /*remove the writespace in the front*/
        auto space_begin = name.find_first_not_of(" ");
        name = name.substr(space_begin, name.size() - space_begin);
        /*remove the writespace in the back*/
        auto space_end = name.find_last_not_of(" ");
        name = name.substr(0, space_end + 1);

        /*remove the type info*/
        auto type_end = name.find(" ");
        name = name.substr(type_end + 1, name.size() - type_end);
    }

    assert(sizeof...(instance) == template_paramter.size());

    /*get the define macro*/
    std::string macro;
    auto lambda = [&](std::string name, std::string value) {
        auto pos = KernelSource.find(name);
        macro += "#define " + name + " " + value + "\n";
    };

    auto i = 0;
    (lambda(template_paramter[i++], instance), ...);

    /*remove the tempalte decalartion*/
    template_begin = KernelSource.rfind("template", pos);
    KernelSource.replace(template_begin, template_end - template_begin + 1,
                         macro);
}