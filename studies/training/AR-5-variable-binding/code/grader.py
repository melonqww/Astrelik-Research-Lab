"""Interpret a small arithmetic AST, never execute generated Python code."""
import ast

def check_function(source,name,arguments,examples):
    if len(source)>16000:return {'syntax':False,'supported':False,'correct':False}
    try:tree=ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):return {'syntax':False,'supported':False,'correct':False}
    def result(supported,correct=False):return dict(syntax=True,supported=supported,correct=correct)
    if len(tree.body)!=1:return result(False)
    f=tree.body[0]
    if not isinstance(f,ast.FunctionDef) or f.name!=name or f.decorator_list or f.returns:return result(False)
    a=f.args
    if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults or a.kw_defaults:return result(False)
    if [x.arg for x in a.args]!=arguments or any(x.annotation for x in a.args):return result(False)
    if len(f.body)!=1 or not isinstance(f.body[0],ast.Return):return result(False)
    def value(node,env,depth=0):
        if depth>12:raise ValueError('AST depth')
        if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<=1000:return node.value
        if isinstance(node,ast.Name) and node.id in env:return env[node.id]
        if isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult):
            left=value(node.left,env,depth+1);right=value(node.right,env,depth+1)
            if type(node.op) is ast.Add:answer=left+right
            elif type(node.op) is ast.Sub:answer=left-right
            else:answer=left*right
            if abs(answer)>10**12:raise ValueError('bounded arithmetic')
            return answer
        if isinstance(node,ast.UnaryOp) and type(node.op) in (ast.UAdd,ast.USub):
            x=value(node.operand,env,depth+1);return x if type(node.op) is ast.UAdd else -x
        raise ValueError('unsupported expression')
    try:answers=[value(f.body[0].value,dict(zip(arguments,x))) for x,_ in examples]
    except (ValueError,RecursionError):return result(False)
    return result(True,answers==[y for _,y in examples])

def qualify():
    tests=[([1,2],3),([-5,3],-2),([0,0],0)]
    assert check_function('def add(a,b):\n return a+b','add',['a','b'],tests)['correct']
    wrong=check_function('def add(a,b):\n return a-b','add',['a','b'],tests)
    assert wrong==dict(syntax=True,supported=True,correct=False)
    for text in ['def add(a,b):\n return __import__("os").system("never")','import os\ndef add(a,b):\n return a+b','def add(a,b):\n while True: pass','def add(a,b):\n return a**999999999']:
        assert not check_function(text,'add',['a','b'],tests)['supported']
    assert not check_function('def add(a,b):\n return (','add',['a','b'],tests)['syntax']
    return 7
