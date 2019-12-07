#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<class OT, class PARAMS>
struct object_pool
{
	typedef std::unique_ptr<OT>(*factory_type)(const PARAMS &params);
	typedef std::map<std::string, factory_type> registry_type;

	factory_type get(const char *name)
	{
		typename registry_type::iterator it = registry.find(name);
		return it != registry.end() ? it->second : nullptr;
	}

	void register_factory(const char *name, factory_type factory){
		registry[name] = factory;
	}

	template<class T>
	static std::unique_ptr<OT> factory(const PARAMS &params){
		return std::unique_ptr<OT>(new T(params));
	}

	//template<class T>
	//struct registrator {
	//	registrator(const char *name){
	//		register_factory(name, factory<T>);
	//	}
	//};

private:
	registry_type registry;
};

_KIAM_MATH_END
