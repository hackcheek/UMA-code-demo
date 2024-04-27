from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.PrintCollector import PrintCollector


MAX_ITER_LEN = 500  # Max num of iterations


class MaxCountIter:
    def __init__(self, dataset, max_count):
        self.i = iter(dataset)
        self.left = max_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.left > 0:
            self.left -= 1
            return next(self.i)
        else:
            raise StopIteration()

def _getiter(ob):
    return MaxCountIter(ob, MAX_ITER_LEN)


def execute_user_code_async(user_code, *args, **kwargs):
    """ Executed user code in restricted env
        Args:
            user_code(str) - String containing the unsafe code
            *args, **kwargs - arguments passed to the user function
        Return:
            Return value of the user_func
    """

    def _apply(f, *a, **kw):
        return f(*a, **kw)

    try:
        # This is the variables we allow user code to see. @result will contain return value.
        restricted_locals = {
            "result": None,
            "args": args,
            "kwargs": kwargs,
        }

        # If you want the user to be able to use some of your functions inside his code,
        # you should add this function to this dictionary.
        # By default many standard actions are disabled. Here I add _apply_ to be able to access
        # args and kwargs and _getitem_ to be able to use arrays. Just think before you add
        # something else. I am not saying you shouldn't do it. You should understand what you
        # are doing thats all.
        restricted_globals = {
            "__builtins__": safe_builtins,
            "__import__": __import__,
            "_getitem_": default_guarded_getitem,
            "_getiter_": _getiter,
            "_apply_": _apply,
            "_print_": PrintCollector,
        }

        # Compile the user code
        byte_code = compile_restricted(user_code, filename="<user_code>", mode="exec")

        # Run it
        exec(byte_code, restricted_globals, restricted_locals)

        # async def _exec():
        #     # Make an async function with the code and `exec` it
        #     exec(
        #         f'async def __FWAI_EXECUTION(): ' +
        #         ''.join(f'\n {l}' for l in code.split('\n'))
        #     )

        #     # Get `__ex` from local variables, call it and return the result
        #     return await locals()['__FWAI_EXECUTION']()
        # asyncio.run(_exec())

        # User code has modified result inside restricted_locals. Return it.
        return restricted_locals["result"]

    except SyntaxError as err:
        # Do whaever you want if the user has code that does not compile
        raise
    except Exception as err:
        # The code did something that is not allowed. Add some nasty punishment to the user here.
        raise RuntimeError(f"For safety we decided not to run your code:\n{str(err)}")

