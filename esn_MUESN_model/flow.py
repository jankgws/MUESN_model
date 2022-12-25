import numpy as np

import copy as _copy
import tempfile as _tempfile
import pickle as _cPickle
import traceback as _traceback
import warnings as _warnings                    
from builtins import object
from inspect import getargspec as getargs

class FlowException(mdp.MDPException):
    """Base class for exceptions in Flow subclasses."""
    pass

class Flow(object):
    """
    フロー」とは、学習され、実行される一連のノードのことである。
    より複雑なアルゴリズムを形成するために一緒に実行されるノードのシーケンスです。 入力データは最初のノードに送られ
    入力データは最初のノードに送られ、シーケンスに沿った後続のノードで順次処理される。
    処理される。

    一連のノードを手動で処理するのとは対照的に，フローを使用すると，次のような利点がある．
    明確な利点がある。一般的なフローの実装では，学習（教師あり学習や複数の学習段階を含む）が自動化される．
    学習（教師あり学習や複数の学習段階を含む）。
    実行、および（定義されていれば）逆実行を自動化する。

    クラッシュリカバリはオプションで利用可能である。
    のサブクラスである．基本フロークラスのサブクラス
    基本フロークラスのサブクラス（'CheckpointFlow'）により、ユーザが提供するチェックポイント関数
    各フェーズの終了時に、ユーザが提供するチェックポイント関数を実行することができます。
    の内部構造を保存し、後で解析することができる。
    フローオブジェクトはPythonのコンテナである。ほとんどの組み込みの'list'
    メソッドが利用できる。Flow」は、対応する「save」と「copy」を使って保存したりコピーしたりすることができる。
    Flow」は、対応する「save」と「copy」メソッドを用いて保存またはコピーすることができる。
    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        """
        キーワード引数。

        flow -- ノードのリスト
        crash_recovery -- クラッシュリカバリーモードの設定(または設定しない) (障害発生時にノード
                          を設定する(しない)。
        verbose -- True の場合、基本的な進捗情報を表示します。
        """
        self._check_nodes_consistency(flow)
        self.flow = flow
        self.verbose = verbose
        self.set_crash_recovery(crash_recovery)

    def _propagate_exception(self, except_, nodenr):
        # capture exception. the traceback of the error is printed and a
        # new exception, containing the identity of the node in the flow
        # is raised. Allow crash recovery.
        (etype, val, tb) = _sys.exc_info()
        prev = ''.join(_traceback.format_exception(except_.__class__,
                                                   except_, tb))
        act = "\n! Exception in node #%d (%s):\n" % (nodenr,
                                                     str(self.flow[nodenr]))
        errstr = ''.join(
            ('\n', 40*'-', act, 'Node Traceback:\n', prev, 40*'-'))
        raise FlowExceptionCR(errstr, self, except_)

    def _train_node(self, data_iterable, nodenr):
        """フロー内の1つのノードを学習させる。

        nodenr -- フローの中のノードのインデックス
        """
        node = self.flow[nodenr]
        if (data_iterable is not None) and (not node.is_trainable()):
            # attempted to train a node although it is not trainable.
            # raise a warning and continue with the next node.
            # wrnstr = "\n! Node %d is not trainable" % nodenr + \
            #        "\nYou probably need a 'None' iterable for"+\
            #         " this node. Continuing anyway."
            #_warnings.warn(wrnstr, mdp.MDPWarning)
            return
        elif (data_iterable is None) and node.is_training():
            # None instead of iterable is passed to a training node
            err_str = ("\n! Node %d is training"
                       " but instead of iterable received 'None'." % nodenr)
            raise FlowException(err_str)
        elif (data_iterable is None) and (not node.is_trainable()):
            # skip training if node is not trainable
            return

        try:
            train_arg_keys = self._get_required_train_args(node)
            train_args_needed = bool(len(train_arg_keys))
            # We leave the last training phase open for the
            # CheckpointFlow class.
            # Checkpoint functions must close it explicitly if needed!
            # Note that the last training_phase is closed
            # automatically when the node is executed.
            while True:
                empty_iterator = True
                for x in data_iterable:
                    empty_iterator = False
                    # the arguments following the first are passed only to the
                    # currently trained node, allowing the implementation of
                    # supervised nodes
                    if (type(x) is tuple) or (type(x) is list):
                        arg = x[1:]
                        x = x[0]
                    else:
                        arg = ()
                    # check if the required number of arguments was given
                    if train_args_needed:
                        if len(train_arg_keys) != len(arg):
                            err = ("Wrong number of arguments provided by " +
                                   "the iterable for node #%d " % nodenr +
                                   "(%d needed, %d given).\n" %
                                   (len(train_arg_keys), len(arg)) +
                                   "List of required argument keys: " +
                                   str(train_arg_keys))
                            raise FlowException(err)
                    # filter x through the previous nodes
                    if nodenr > 0:
                        x = self._execute_seq(x, nodenr-1)
                    # train current node
                    node.train(x, *arg)
                if empty_iterator:
                    if node.get_current_train_phase() == 1:
                        err_str = ("The training data iteration for node "
                                   "no. %d could not be repeated for the "
                                   "second training phase, you probably "
                                   "provided an iterator instead of an "
                                   "iterable." % (nodenr+1))
                        raise FlowException(err_str)
                    else:
                        err_str = ("The training data iterator for node "
                                   "no. %d is empty." % (nodenr+1))
                        raise FlowException(err_str)
                self._stop_training_hook()
                if node.get_remaining_train_phase() > 1:
                    # close the previous training phase
                    node.stop_training()
                else:
                    break
        except mdp.TrainingFinishedException as e:
            # attempted to train a node although its training phase is already
            # finished. raise a warning and continue with the next node.
            wrnstr = ("\n! Node %d training phase already finished"
                      " Continuing anyway." % nodenr)
            _warnings.warn(wrnstr, mdp.MDPWarning)
        except FlowExceptionCR as e:
            # this exception was already propagated,
            # probably during the execution  of a node upstream in the flow
            (exc_type, val) = _sys.exc_info()[:2]
            prev = ''.join(_traceback.format_exception_only(e.__class__, e))
            prev = prev[prev.find('\n')+1:]
            act = "\nWhile training node #%d (%s):\n" % (nodenr,
                                                         str(self.flow[nodenr]))
            err_str = ''.join(('\n', 40*'=', act, prev, 40*'='))
            raise FlowException(err_str)
        except Exception as e:
            # capture any other exception occured during training.
            self._propagate_exception(e, nodenr)

    def _stop_training_hook(self):
        """Hook method that is called before stop_training is called."""
        pass

    @staticmethod
    def _get_required_train_args(node):
        """Return arguments in addition to self and x for node.train.

        Argumentes that have a default value are ignored.
        """
        train_arg_spec = getargs(node._train)
        train_arg_keys = train_arg_spec[0][2:]  # ignore self, x
        if train_arg_spec[3]:
            # subtract arguments with a default value
            train_arg_keys = train_arg_keys[:-len(train_arg_spec[3])]
        return train_arg_keys

    def _train_check_iterables(self, data_iterables):
        """Return the data iterables after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """
        # verifies that the number of iterables matches that of
        # the signal nodes and multiplies them if needed.
        flow = self.flow

        # if a single array is given wrap it in a list of lists,
        # note that a list of 2d arrays is not valid
        if isinstance(data_iterables, numx.ndarray):
            data_iterables = [[data_iterables]] * len(flow)

        if not isinstance(data_iterables, list):
            err_str = ("'data_iterables' must be either a list of "
                       "iterables or an array, and not %s" %
                       type(data_iterables))
            raise FlowException(err_str)

        # check that all elements are iterable
        for i, iterable in enumerate(data_iterables):
            if (iterable is not None) and (not hasattr(iterable, '__iter__')):
                err = ("Element number %d in the data_iterables"
                       " list is not an iterable." % i)
                raise FlowException(err)

        # check that the number of data_iterables is correct
        if len(data_iterables) != len(flow):
            err_str = ("%d data iterables specified,"
                       " %d needed" % (len(data_iterables), len(flow)))
            raise FlowException(err_str)

        return data_iterables

    def _close_last_node(self):
        if self.verbose:
            print("Close the training phase of the last node")
        try:
            self.flow[-1].stop_training()
        except mdp.TrainingFinishedException:
            pass
        except Exception as e:
            self._propagate_exception(e, len(self.flow)-1)

    def set_crash_recovery(self, state=True):
        """Set crash recovery capabilities.

        When a node raises an Exception during training, execution, or
        inverse execution that the flow is unable to handle, a FlowExceptionCR
        is raised. If crash recovery is set, a crash dump of the flow
        instance is saved for later inspection. The original exception
        can be found as the 'parent_exception' attribute of the
        FlowExceptionCR instance.

        - If 'state' = False, disable crash recovery.
        - If 'state' is a string, the crash dump is saved on a file
          with that name.
        - If 'state' = True, the crash dump is saved on a file created by
          the tempfile module.
        """
        self._crash_recovery = state

    def train(self, data_iterables):
        """Train all trainable nodes in the flow.

        'data_iterables' is a list of iterables, one for each node in the flow.
        The iterators returned by the iterables must return data arrays that
        are then used for the node training (so the data arrays are the 'x' for
        the nodes). Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained, so the data dimension
        must match the input dimension of the first node.

        If a node has only a single training phase then instead of an iterable
        you can alternatively provide an iterator (including generator-type
        iterators). For nodes with multiple training phases this is not
        possible, since the iterator cannot be restarted after the first
        iteration. For more information on iterators and iterables see
        http://docs.python.org/library/stdtypes.html#iterator-types .

        In the special case that 'data_iterables' is one single array,
        it is used as the data array 'x' for all nodes and training phases.

        Instead of a data array 'x' the iterators can also return a list or
        tuple, where the first entry is 'x' and the following are args for the
        training of the node (e.g. for supervised training).
        """

        data_iterables = self._train_check_iterables(data_iterables)

        # train each Node successively
        for i in range(len(self.flow)):
            if self.verbose:
                print("Training node #%d (%s)" % (i, str(self.flow[i])))
            self._train_node(data_iterables[i], i)
            if self.verbose:
                print("Training finished")

        self._close_last_node()

    def _execute_seq(self, x, nodenr=None):
        # Filters input data 'x' through the nodes 0..'node_nr' included
        flow = self.flow
        if nodenr is None:
            nodenr = len(flow)-1
        for i in range(nodenr+1):
            try:
                x = flow[i].execute(x)
            except Exception as e:
                self._propagate_exception(e, i)
        return x

    def execute(self, iterable, nodenr=None):
        """Process the data through all nodes in the flow.

        'iterable' is an iterable or iterator (note that a list is also an
        iterable), which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.

        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'. This is equivalent to 'flow[:nodenr+1](iterable)'.
        """
        if isinstance(iterable, numx.ndarray):
            return self._execute_seq(iterable, nodenr)
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._execute_seq(x, nodenr))
        if empty_iterator:
            errstr = ("The execute data iterator is empty.")
            raise FlowException(errstr)
        return numx.concatenate(res)

    def _inverse_seq(self, x):
        # Successively invert input data 'x' through all nodes backwards
        flow = self.flow
        for i in range(len(flow)-1, -1, -1):
            try:
                x = flow[i].inverse(x)
            except Exception as e:
                self._propagate_exception(e, i)
        return x

    def inverse(self, iterable):
        """Process the data through all nodes in the flow backwards
        (starting from the last node up to the first node) by calling the
        inverse function of each node. Of course, all nodes in the
        flow must be invertible.

        'iterable' is an iterable or iterator  (note that a list is also an
        iterable), which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.

        Note that this is _not_ equivalent to 'flow[::-1](iterable)',
        which also executes the flow backwards but calls the 'execute'
        function of each node."""

        if isinstance(iterable, numx.ndarray):
            return self._inverse_seq(iterable)
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._inverse_seq(x))
        if empty_iterator:
            errstr = ("The inverse data iterator is empty.")
            raise FlowException(errstr)
        return numx.concatenate(res)

    def copy(self, protocol=None):
        """Return a deep copy of the flow.

        The protocol parameter should not be used.
        """
        if protocol is not None:
            _warnings.warn("protocol parameter to copy() is ignored",
                           mdp.MDPDeprecationWarning, stacklevel=2)
        return _copy.deepcopy(self)

    def save(self, filename, protocol=-1):
        """Save a pickled serialization of the flow to 'filename'.
        If 'filename' is None, return a string.

        Note: the pickled Flow is not guaranteed to be upward or
        backward compatible."""
        if filename is None:
            return _cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            mode = 'w' if protocol == 0 else 'wb'
            with open(filename, mode) as flh:
                _cPickle.dump(self, flh, protocol)

    def __call__(self, iterable, nodenr=None):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.execute(iterable, nodenr=nodenr)

    # string representation

    def __str__(self):
        nodes = ', '.join([str(x) for x in self.flow])
        return '['+nodes+']'

    def __repr__(self):
        # this should look like a valid Python expression that
        # could be used to recreate an object with the same value
        # eval(repr(object)) == object
        name = type(self).__name__
        pad = len(name)+2
        sep = ',\n'+' '*pad
        nodes = sep.join([repr(x) for x in self.flow])
        return '%s([%s])' % (name, nodes)

    # private container methods

    def __len__(self):
        return len(self.flow)

    def _check_dimension_consistency(self, out, inp):
        """Raise ValueError when both dimensions are set and different."""
        if ((out and inp) is not None) and out != inp:
            errstr = "dimensions mismatch: %d != %d" % (out, inp)
            raise ValueError(errstr)

    def _check_nodes_consistency(self, flow=None):
        """Check the dimension consistency of a list of nodes."""
        if flow is None:
            flow = self.flow
        len_flow = len(flow)
        for i in range(1, len_flow):
            out = flow[i-1].output_dim
            inp = flow[i].input_dim
            self._check_dimension_consistency(out, inp)

    def _check_value_type_isnode(self, value):
        if not isinstance(value, mdp.Node):
            raise TypeError("flow item must be Node instance")

    def __getitem__(self, key):
        if isinstance(key, slice):
            flow_slice = self.flow[key]
            self._check_nodes_consistency(flow_slice)
            return self.__class__(flow_slice)
        else:
            return self.flow[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            [self._check_value_type_isnode(item) for item in value]
        else:
            self._check_value_type_isnode(value)

        # make a copy of list
        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __delitem__(self, key):
        # make a copy of list
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __contains__(self, item):
        return self.flow.__contains__(item)

    def __iter__(self):
        return self.flow.__iter__()

    def __add__(self, other):
        # append other to self
        if isinstance(other, Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.Node):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, Flow):
            self.flow += other.flow
        elif isinstance(other, mdp.Node):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)
        self._check_nodes_consistency(self.flow)
        return self

    # public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, Flow):
            err_str = ('can only concatenate flow'
                       ' (not \'%s\') to flow' % (type(x).__name__))
            raise TypeError(err_str)
        self[len(self):len(self)] = x

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]

    def pop(self, i=-1):
        """flow.pop([index]) -> node -- remove and return node at index
        (default last)"""
        x = self[i]
        del self[i]
        return x
