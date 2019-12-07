#pragma once

#define RT_EXPRESSION_INCLUDED

#include <map>
#include <memory>

#include "math_def.h"

_KIAM_MATH_BEGIN

struct rt_expression
{
	virtual ~rt_expression() {}
	virtual double calc(std::map<std::string, double> const& vars) const = 0;
};

struct rt_constant : rt_expression
{
	rt_constant(double value) : value(value){}

	double calc(std::map<std::string, double> const&) const override;

private:
	double const value;
};

struct rt_variable : rt_expression
{
	rt_variable(const char *name) : name(name){}
	rt_variable(const std::string &name) : name(name){}

	double calc(std::map<std::string, double> const& vars) const override;

private:
	std::string const name;
};

struct rt_function_call : rt_expression
{
	rt_function_call(const char *func_name, const std::shared_ptr<const rt_expression> &expr) : func_name(func_name), expr(expr){}

	double calc(std::map<std::string, double> const& vars) const override;

private:
	std::string const func_name;
	std::shared_ptr<const rt_expression> const expr;
};

struct rt_unary_expression : rt_expression
{
	rt_unary_expression(char op, std::shared_ptr<const rt_expression> const& expr) : op(op), expr(expr){}

	double calc(std::map<std::string, double> const& vars) const override;

private:
	char const op;
	std::shared_ptr<const rt_expression> const expr;
};

struct rt_binary_expression : rt_expression
{
	rt_binary_expression(std::shared_ptr<const rt_expression> const& expr1, char op, std::shared_ptr<const rt_expression> const& expr2) :
		expr1(expr1), op(op), expr2(expr2){}

	double calc(std::map<std::string, double> const& vars) const override;

private:
	std::shared_ptr<const rt_expression> const expr1;
	char const op;
	std::shared_ptr<const rt_expression> const expr2;
};

struct rt_condition
{
	virtual ~rt_condition() {}
	virtual bool calc(std::map<std::string, double> const& vars) const = 0;
};

struct rt_not_condition : rt_condition
{
	rt_not_condition(std::shared_ptr<const rt_condition> const& cond) : cond(cond){}

	bool calc(std::map<std::string, double> const& vars) const override;

private:
	std::shared_ptr<const rt_condition> const cond;
};

struct rt_binary_condition : rt_condition
{
	rt_binary_condition(std::shared_ptr<const rt_condition> const& cond1, std::string const& op, std::shared_ptr<const rt_condition> const& cond2) :
		cond1(cond1), op(op), cond2(cond2){}

	bool calc(std::map<std::string, double> const& vars) const override;

private:
	std::shared_ptr<const rt_condition> const cond1;
	std::string const op;
	std::shared_ptr<const rt_condition> const cond2;
};

struct rt_compare_condition : rt_condition
{
	rt_compare_condition(std::shared_ptr<const rt_expression> const& expr1, std::string const& op, std::shared_ptr<const rt_expression> const& expr2) :
		expr1(expr1), op(op), expr2(expr2){
	}

	bool calc(std::map<std::string, double> const& vars) const override;

private:
	std::shared_ptr<const rt_expression> const expr1;
	std::string const op;
	std::shared_ptr<const rt_expression> const expr2;
};

struct rt_logical_expression : rt_expression
{
	rt_logical_expression(std::shared_ptr<const rt_condition> const& cond, std::shared_ptr<const rt_expression> const& expr1, std::shared_ptr<const rt_expression> const& expr2) :
		cond(cond), expr1(expr1), expr2(expr2){}

	double calc(std::map<std::string, double> const& vars) const override;

private:
	std::shared_ptr<const rt_condition> const cond;
	std::shared_ptr<const rt_expression> const expr1;
	std::shared_ptr<const rt_expression> const expr2;
};

std::shared_ptr<const rt_expression> parse_rt_expression(std::string const& sexpr);

_KIAM_MATH_END
