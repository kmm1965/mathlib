#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<class OT, typename... ARGS>
struct object_pool
{
    typedef std::unique_ptr<OT>(*factory_type)(ARGS...);
    typedef std::map<std::string, factory_type> registry_type;

    factory_type get(const char *name)
    {
        typename registry_type::iterator it = registry.find(name);
        return it != std::end(registry) ? it->second : nullptr;
    }

    void register_factory(const char *name, factory_type factory){
        registry[name] = factory;
    }

    template<class T>
    static std::unique_ptr<OT> factory(ARGS... args){
        return std::unique_ptr<OT>(new T(args...));
    }

private:
    registry_type registry;
};

_KIAM_MATH_END
