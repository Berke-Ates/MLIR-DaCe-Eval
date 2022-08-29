import sys

# sys.path.insert(0, "/home/xdb/dace")

import time
import dace

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

# locDom = new Domain(1, 0, 0, 0, 30,
#                     1, 11, 1, 1);

m_p = dace.ndarray(shape=(27000, ), dtype=dace.float64)
m_q = dace.ndarray(shape=(27000, ), dtype=dace.float64)

m_x = dace.ndarray(shape=(29791, ), dtype=dace.float64)  # Has Values
m_y = dace.ndarray(shape=(29791, ), dtype=dace.float64)  # Has Values
m_z = dace.ndarray(shape=(29791, ), dtype=dace.float64)  # Has Values

m_fx = dace.ndarray(shape=(29791, ), dtype=dace.float64)
m_fy = dace.ndarray(shape=(29791, ), dtype=dace.float64)
m_fz = dace.ndarray(shape=(29791, ), dtype=dace.float64)

m_nodelist = dace.ndarray(shape=(216000, ), dtype=dace.int64)  # Has Values
m_nodeElemStart = dace.ndarray(shape=(1, ), dtype=dace.int64)  # Zero?
m_nodeElemCornerList = dace.ndarray(shape=(1, ), dtype=dace.int64)  # Zero?
m_numElem = dace.ndarray(shape=(1, ), dtype=dace.int64)  # Is 27000

m_volo = dace.ndarray(shape=(27000, ), dtype=dace.float64)  # Has Values
m_v = dace.ndarray(shape=(27000, ), dtype=dace.float64)  # Is all 1
m_ss = dace.ndarray(shape=(27000, ), dtype=dace.float64)
m_elemMass = dace.ndarray(shape=(27000, ), dtype=dace.float64)  # Has Values

m_xd = dace.ndarray(shape=(29791, ), dtype=dace.float64)
m_yd = dace.ndarray(shape=(29791, ), dtype=dace.float64)
m_zd = dace.ndarray(shape=(29791, ), dtype=dace.float64)

m_numNode = dace.ndarray(shape=(1, ), dtype=dace.int64)  # 29791
m_hgcoef = dace.ndarray(shape=(1, ), dtype=dace.float64)  # 3.000000

start_time = time.time()
obj(
    _arg0=m_p,
    s_0=27000,
    _arg1=m_q,
    s_1=27000,
    _arg2=m_x,
    s_2=29791,
    _arg3=m_y,
    s_3=29791,
    _arg4=m_z,
    s_4=29791,
    _arg5=m_fx,
    s_5=29791,
    _arg6=m_fy,
    s_6=29791,
    _arg7=m_fz,
    s_7=29791,
    _arg8=m_nodelist,
    s_8=216000,
    _arg9=m_nodeElemStart,
    s_9=1,
    _arg10=m_nodeElemCornerList,
    s_10=1,
    _arg11=m_numElem,
    s_11=1,
    _arg12=m_volo,
    s_12=27000,
    _arg13=m_v,
    s_13=27000,
    _arg14=m_ss,
    s_14=27000,
    _arg15=m_elemMass,
    s_15=27000,
    _arg16=m_xd,
    s_16=29791,
    _arg17=m_yd,
    s_17=29791,
    _arg18=m_zd,
    s_18=29791,
    _arg19=m_numNode,
    s_19=1,
    _arg20=m_hgcoef,
    s_20=1,
)
print("%d" % int((time.time() - start_time) * 1000))
