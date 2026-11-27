import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import io
import base64
import re
import csv
from copy import copy
import openpyxl

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Extractor Facturas AI", layout="wide")

# Modelos disponibles (nombre en pantalla -> id de la API)
MODELOS = {
    "Económico y rápido · Gemini 3.5 Flash-Lite (recomendado)": "gemini-3.5-flash-lite",
    "Más preciso · Gemini 3.8 Flash": "gemini-3.8-flash",
}

# Plantilla del Excel de salida (entrada de apuntes), incluida dentro de este archivo.
PLANTILLA_B64 = (
    "UEsDBBQAAAAIAGSFe11Gx01IlQAAAM0AAAAQAAAAZG9jUHJvcHMvYXBwLnhtbE3PTQvCMAwG4L9SdreZih6kDkQ9ip68zy51hbYp"
    "bYT67+0EP255ecgboi6JIia2mEXxLuRtMzLHDUDWI/o+y8qhiqHke64x3YGMsRoPpB8eA8OibdeAhTEMOMzit7Dp1C5GZ3XPlkJ3"
    "sjpRJsPiWDQ6sScfq9wcChDneiU+ixNLOZcrBf+LU8sVU57mym/8ZAW/B7oXUEsDBBQAAAAIAGSFe13WSL5HcgEAAGcDAAARAAAA"
    "ZG9jUHJvcHMvY29yZS54bWzFk01OwzAQha8SZZ86P1UBK0TiR7ChUqWCQOwGZ9paJLZlTyntnTgFF8MxTaCCDSt28fi9z8/jSSkM"
    "F9rizGqDliS66LVtlOPCnMYrIsMZc2KFLbiRVyi/udC2BfJLu2QGxDMskeVpOmEtEtRAwDpgYgZiXJW14MIikLZ7fC0GvFnbJsBq"
    "wbDBFhU5lo0yFlfXWtEOmpJ9+QOLJDX4J5L/tD5cVGMEZq0IXYAGUEA2oJZrf5U/UpO7eQD19o5FaFv3mRfrgReqv0LDDov3ylcn"
    "B9VmsxltiqDzHc7Yw/RmHh4jkcoRKIHe5SSnrcHTuD/5vri4vL2KK28pkjRP8uPbbMKzgo+PHruwB/m+Are6lgv5z4nzSZJlSX7U"
    "JR4XPD3+lrgPWJV+aBtwNN0XzrfdpPhB0dEULL2/KdxFLDoH+wT1WkVnDp22729Qsp/ODmbxRTqpVZUGxbDsD5pZqbpeHXZ0fMLT"
    "k8eB2YtC4fCfqj4AUEsDBBQAAAAIAGSFe132sPGCHgIAANEIAAATAAAAeGwvdGhlbWUvdGhlbWUxLnhtbN2VTW/bMAyG7/sVgu6r"
    "4rgJ0iBOMSwLdiiwQ7bdGZm21UiyIant8u+nyE7ir6HDMGDofIlIPXxFioy9uv+hJHlGY0WpExrdTChBzctU6Dyh375u3y8osQ50"
    "CrLUmNAjWnq/freCpStQIfHh2i4hoYVz1ZIxy70b7E1ZofZ7WWkUOG+anKUGXryskmw6mcyZAqFpE29+J77MMsFxU/InhdrVIgYl"
    "OJ+6LURlKdGgfI5fAkjX5yQ/STxF2JODS7PjIfOafRB7g62A9BCdfqzJ9x+lIc8gEzoJD2XrFbsA0g25LDwN1wDpYfqa3rTWG3I9"
    "vQAA576U4dnRAuJJ3LAtqF6O5BDP76DLt/TjAQ9xjD39+MrfDviFp3v6t1d+NuD53R2/3EkLqpfzEX4aRdjhA1RIoQ+jN45n+oJk"
    "pfw8is9mESz2DX6lWGt86njtOsPUmiMFj6XZeiA018+oJu5YYQbccx+MAElJJRwvtqCEPPoUKeEFGIvON/N0NCwRWjEbfITvT2QH"
    "2r4eye2fRbJe4kroN1rFNXHWblRom2obQsqdO0p8sKFIW0qRbr0zGAG7jEVV+CUNiped2uoE/XMFNixL6q5FXhI6j2enq4PKv2l8"
    "b/1SVWlCrc4pAZn7zwF3JgxzZazbgC3qFMJJdYeUcGia95N+m8qsfzmYZcjdLzxX0+/VIqO7fx9mY5nt8+3/Ob/9wljnb8sGH/az"
    "Z/0TUEsDBBQAAAAIAGSFe13npPSMrBgAADW7AAAYAAAAeGwvd29ya3NoZWV0cy9zaGVldDEueG1snZ1tb9vGmob/iuEFFrs8gG2+"
    "k9kkQGomnGHdptvktMB+U20mFmpbXllJTvvrd6R5mOQxZ66Emw9JxPuFEm/Zbi+04tNPm+2fD9fjuDv61+3N3cOz4+vd7v7J6enD"
    "5fV4u3o42dyPd055t9nernbu4fb96cP9dlxdHUK3N6fZ2Vl1erta3x0/f3o49sv26N36Zjduf9pcjc+Oz9zxzYfdzfpudMrDh9vb"
    "1favH8abzadnx+nxdODX9fvr3f7A6fOn96v345tx98/7Q9Pu7eYXd2Bf5LRTOcXzp1fr2/HuYb25O9qO754dv0ifXKRnB8/B8tt6"
    "/PTw1d+PHq43n165V/HhZvWwLzsc6Lfrqwv3zB78c3FHft18Ot/cGPcCx+2Xo/8zbjf+0Xb/RN9uLsZ3u0PLbvXHm/FmvNyNV5/d"
    "r/3LffPX7R+bGx+7Gt+tPtzs9udz9Zvt4eBH97yeHd/tL+2Na9rc71vPx5ub/cs5PrrcG62rrYrjo783m9s3l6sbdyH2L/PL458P"
    "8cdH95fsYvWXu/C/HU5yUPdb/7HZ/Lk/tO/dT/NwePL7y3i/unPl8iyOj1bu6Mfxy7P58thHjx7+Vy78l132xV//fRrg1eHd4+b8"
    "Y/Uwutf/+/pqd/3suPl8Xb4cS9OTs7Komrr8LLpJzCjvj+xkL/zt5pgOuWcib6+L8eN48+v+fXV4G7jL93D4/eiTVJ+dZO6Ulx8e"
    "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"
    "OSmjuseHVfd/HmaVv0R2fRTf1yVTNjxtKGEiCb/u6cP1OO661W71/OntuH0/no83Nw9Hl5sPd+7l7Pf8fPRoO77b7/OkS49P58fz"
    "J10eOP4iffJD0J896bL98dMvp33+9H67vtu9vt+tN3cPR9eb7frvzd1udXM+3u3G7Xj17Ni9vo/jdre+fHTwelxdre/ePxwevN+u"
    "ry7Wd+OjR2/GwzvVnfN+9X78abV9v3ZnuRnfucNnJ3Wzf1tv/ft5erjb3O8ftE1xltWff7k33x+bnXv7h7X9cxm3e61M0yZNz7K8"
    "yrKz/f0H3m02u7Akz8o9xw/3R+6Fuxe32l+GZ8c3q7urh8vV/Xh8dO9+375Z/z36rzh3EUb//969W+/ebr76Yjw8/n19tbs+PNw3"
    "v94entTV5tPd2+vx7rW7jO5l3Kwu/3xxd/X79Xo3Hq7W1XZ1uB7HX13+7n797Dg/++rafzlyublf7y/04bqeftps/zy8o57/H1BL"
    "AwQUAAAACABkhXtd4Y55p00TAACIlQAAGAAAAHhsL3dvcmtzaGVldHMvc2hlZXQyLnhtbJ2dX28bt5tGv4rhBRa7XMD2kJx/3iSA"
    "EyYkg7QukvxaYO9UexwLlSWvpCRNP/1SIic1Hc1xub1Io3nf88zo0djIuRk9+7pa/7G5HYbt0Z93i+Xm+fHtdnt/fnq6ubod7mab"
    "k9X9sAyTm9X6brYNL9efTjf362F2vYfuFqfy7Kw5vZvNl8cvnu2P/bI+upkvtsP6p9X18Pz4LBxffd4u5sshTDaf7+5m628vh8Xq"
    "6/Pj6ng88H7+6Xa7O3D64tn97NPwYdj+636ftP24+iUc2AWF2Wk6xYtn1/O7YbmZr5ZH6+Hm+fFFdf6273Yr+41f58PXzYO/H21u"
    "V1/fhDfxeTHb7LL2B+x6fv0uXNgmXko48n719dVq4cL7G9Z/H/2fYb2Kr9a76/y4ejfcbPcp29nvH4bFcLUdrr9vX8Z3++Hb3e+r"
    "RcSuh5vZ58V2d74Qv1rvD34J1/X8eLlrdhGSVve71FfDYrF7N8dHV7tFH2IbfXz012p19+Fqtgg9VGdnD17/vMcfH9019m72LfT+"
    "6/4k++nuo/59tfpjd2iXu/tkNvuL37V4P1uG8HQVx0ezcPTLEK/GVNXDA5E92vzvvvjd8PsHs4t++PfxI3izv33C5/n7bDOEBn6b"
    "X29vnx9335v5+1hVnZzVuuna+vswfChuSDeIPNkN/gofyHgoXEq6v94NX4bF+92Ntb9XQoGb/Z9HX1P02YkMp7z6vNmu7sbThTey"
    "/bbrVcvjo7v5cn/obvZnuhkf4vqkfZqWiZaP6P5Etv3D/56OUilKPYrS6qRunsZ1wvXj91H/oxrqhNc/4rp6Gm8S3vzQQ1c/TbeJ"
    "bh+f/Azfukp4l/Du//fWq7PxFoi/dOKdtL+TzWw7e/Fsvfp6FH+I9zfl9zf0/TYNmSFPh5+wq93iRToQFsN0vtz/ftiuw3gegrcv"
    "Xq3u7tezzbPTbTjb7tDpVSJfJbLfk7tfsrvRabiA71ch//FVyIdXMWbF08gnT6P+8WnU9GnUk6fR6TTxBz2eKf762J/p0Klr+f3U"
    "en+gkRM9v3n9yl0caPll4tTU53NpTn55f/nr69fm0Gf0BP3zv/9b1TX/ffTm/cXJAdw8efKfXv/88eK9vzwAv06wnoA/Xn68eDd1"
    "5jf/4G0f2YsPHy8/HKDtU6f2v1we+V8P9e2e+Jw+vH71yl/+fID0T5z05cWH10f+p18Ovd23T7Cv/hXK+vGKs9uzPnB7Tv4g1PGA"
    "+vEH4WUc1e2Bn5HpkUmj/sfR6zhqzn4cvZkOtNOUm6Z8oupw8ObF6/r0P6r/svVp+EfGfz47vXnx7EvY/fKw92zf1yIuP97Nem5K"
    "em6me26me54emWa652a65+lAO025aco3Wc/NvucGes72fSPiMvbclvTcTvfcTvc8PTLtdM/tdM/TgXaactOUb7Oe233PLfSc7ftW"
    "xGXsuSvpuZvuuZvueXpkuumeu+mepwPtNOWmKd9lPXf7njvoOdv3nYjL2HNf0nM/3XM/3fP0yPTTPffTPU8H2mnKTVO+z3ru9z33"
    "0HO273sRl7Hn3T+R/3nRu+2pptPsYNUwM+PsUNlpdrBtyLTAOeD8yKXGq7N95eF/053nSEgQaZ9rr4pqr6D2CmqfnplxdrD2Cmqf"
    "zrTAOeD8yI21V7H2imqv8torkfa5dllUu4TaJdQ+PTPj7GDtEmqfzrTAOeD8yI21y1i7pNplXrsUaZ9rV0W1K6hdQe3TMzPODtau"
    "oPbpTAucA86P3Fi7irUrql3ltSuR9rn2Qw4+XbuG2jXUPj0z4+xg7Rpqn860wDng/MiNtetYu6badV67Fmmfay9yywrksgK7hJmp"
    "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"
    "RNrn2su+EpW+E5W+FJW+FZW+FpW+F5W+GJW+GZW+GvXRd6OmL0fFb0d99PWou+9HfdpSuyJL7cBSO7BUmJkOLLUDS4VMC5wDzne5"
    "pXbRUjuy1BwJCSLtY+19kaX2YKk9WCrMTA+W2oOlQqYFzgHn+9xS+2ipPVlqjoQEkfa59iJL7cFSe7BUmJkeLLUHS4VMC5wDzve5"
    "pfbRUnuy1BwJCSLtc+1FltqDpfZgqTAzPVhqD5YKmRY4B5zvc0vto6X2ZKk5EhJE2ufaiyy1B0vtwVJhZnqw1B4sFTItcA443+eW"
    "2kdL7clScyQkiLTPtRdZag+W2oOlwsz0YKk9WCpkWuAccL7PLbWPltqTpeZISBBpn2svstQeLLUHS4WZ6cFSe7BUyLTAOeB8n1tq"
    "Hy21J0vNkZAg0j7XXmSpPVhqD5YKM9ODpfZgqZBpgXPA+T631D5aak+WmiMhQaR9rr3IUnuw1B4sFWZmnB2sHSwVMi1wDjg/cmPt"
    "0VJ7stQcCQki7XPtRZbag6X2YKkwMz1Yag+WCpkWOAec73NL7aOl9mSpORISRNo/XPvp5nYYtma2nb14djesPw2vhsVic3S1+rwM"
    "V9McPzh6tB5udg/qPX+rjk9/OH4hz1/KA8dfVedvq0PH5fnbQ/sX1fnLQ/sX6vzl/rynf1/mi2f36/lye3m/na+Wm6Pb1Xr+12q5"
    "nS1eDcvtsB6unx+Hxr8M6+386tHB22F2PV9+2uxffFrPr9/Nl8OjVx+G/T0Xznk/+zT8NFt/moezLIabcPjspO12N+g63pnjy+3q"
    "fvei7/SZbL//F26l31fbcCMfnu2uZVjvZnVVdVV1JlUj5dnu8WM3q9X28ChdVbjGz/dH4Y2HNzfb1fD8eDFbXm+uZvfD8dF9+HP9"
    "Yf7XEH92Qgm7v52FN3kz335cPfi52r/+bX69vd2/3EVfrvdXdb36uvx4OywvQ4/hfSxmV39cLK9/u51vh31d1+vZvpDjB/2b+/nu"
    "a3YelP/3kavV/XzX9L7YxfBpdvXNrGdfw4dx9OfdYrk5D+e83W7vz09PN1e3w91sc7K6H5ZhdrNa38224eX60+nq5mZ+NZjV1ee7"
    "8M5P5dlZc7oeFvsKNrfz+034aM7n4bOeLb9tvtwt9nfO19X6j/39/uL/AFBLAwQUAAAACABkhXtdOOrnaaoAAAAKAQAAGAAAAHhs"
    "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"
    "/l7o6WA4Xmjok2/Gnupix7dizx958MVsriFPFbltIC/Ho9Fs/nTJvOiUlqZTjEejJTtF05NDBQWaU2OGIgn5jubUmFkTMytWLN07"
    "QkKBYz41/j1a+GYtfLMWZijyzVqYWRMzK/bNWmihCy2Ce7QIzFoEZi3MUBSYtTCzJmZWHJi10EIXWoT3aBGatQjNWpihKDRrYWZN"
    "zKw4NGuhhS60aN+jRVsaGhot2mYtzFDUNmthZk3MrLht1kILXWjhOveIwZ82dcYR08nxBRYpTCvIF7zJF7z4yNP+0nNuvkRd9y5R"
    "XHOLKEwvihmLFKYXxcybfMGLjzytKFpMilKv/FrPMN2K8a+EwXSfM/6FU7GqIdzrDMWgc2UfeZ1Ya3c7sauxwzA/1NobnWFD56fR"
    "iYW9fk6z3y1okrN5wafb0toRmnyQnKF0hHOGqZrX3zBlyerKuIPxP8m38p5ge74jqK6WuHpd8YjoNoEoqbwXEM1D1XAqFjBXiwZ6"
    "JQw6Sk494pKBP9By3dB1Ha/he57ThN3dEML00Pl6ZF9YUBKkjXiBPTuF4adcoQLbMMoXmC6TDyx7Vl4btLlffptSaWexPg9w3POc"
    "iqTW5JC/7HA+B4Eg6xStfg/y9a9dwsRNjLWmSN2AnIWNigRGFaei6tmyIkUir1n4Lp0unfr/A1BLAwQUAAAACABkhXtdg3FbKkYS"
    "AAA9lgAAGAAAAHhsL3dvcmtzaGVldHMvc2hlZXQ0LnhtbKXdX3PTSJfH8beSylbtDQWJ+n+zQFVI0426GJgC9pmqvfMkgrgmibOO"
    "gWfm1e+xYwEC6ZvZ0VxQuH8+R61j2cznRnryZbX+4/ai6zYH/766vL59enix2dw8Pjq6Pbvorha3j1Y33bUkH1brq8VGXq4/Ht3e"
    "rLvF+a7o6vJIHR+7o6vF8vrw2ZPd2q/rgw/Ly023/mV13j09PJb11afN5fK6k+T209XVYv3n8+5y9eXpYXPYL7xdfrzYbBeOnj25"
    "WXzs3nWb/77Zddq8X/0qC9tGkh3tD/Hsyfnyqru+Xa6uD9bdh6eHJ83j2hzv3rN7y7+W3Zfb7/5+cHux+pLlLD5dLm63zXYLZb08"
    "fyU7u73bi6y8XX05XV2+lBPs1t9W/6dbr+5erbcbfb961X3Y7LpsFr+/6y67s013/rXrm7vTfffn1e+ry7uy8+7D4tPlZns8ab9a"
    "7xY/y76eHl5vR3spnVY3266n3eXl9nQOD862b2ylrTOHB3+tVlfvzhaXMojtaX57/XpX/uPqdmSvFn/K4P+1O8gu3X7Wv69Wf2yX"
    "2vO7j+Z2t/ntGG8W19J8v4vDg4Wsfu7udlPN96/b/Zn+727wkn39XLaNv/97/wHk3dUjH+fvi9tOzv+35fnm4ulh+DqXb2tN8+jY"
    "Ghe8/RrKR/Ky218f6tE2+Es+jn5JdrK/vF51n7vLt9vrancZyPhud38efNm3Pn6k5JBnn243q6v+cHIemz+3UzXq8OBqeb1bulr8"
    "e38tfl9uHvn7q9W+Wv1QHR8pH7//7/5Wet9K/7gR+7fOw+zL7T86D7evdj9UG/3IuvvL/b7c/1AeHgV7f3XYV4cft97gwfW+PO7L"
    "4z8bXHPcXwF3Pzl3F9LuQk6LzeLZk/Xqy8HdN3h3TX49oa9XqfSUfka+XmfbN57sF+SNki6vdz8Om7XES2m8efZ8cX22OmieHG3k"
    "aNulo7N95em+Mu4qtz+x2+hINvB1F+pv70J9v4u+191h1L2H0X/7MHr6MHeRb36OiuYRvVtcnq8OFtfyz8pytX48Mqp2+rh1f1z1"
    "Nfr87PjJ0eeR0zT707z7nbk707tfr92Zjp26VV9P3dwdR0+cRH5x+vJkZOvP76k7fZMePT95ffpm7Aq5p/b1f/5HE9x/HaQ3p49G"
    "ytO+3Ewe+vnbscO+uKfu15MyVpbvPdPX79+e/Hry9v3YZsu91b+8eP3+5G07duiXf3NQ+e3J2LHbe8rfvTg9bd+8Hqms94zq3cmr"
    "9MOGB5ek/f9fkubbJWnvrlH387fi+T7yI1/U6SjdRe745+jFdJSnG5Z9FH+OXk5XtdNR3W/DyuKHZ1U/SPbhC/vk6MP2az/+rXez"
    "RuymR7yPwsiI3fSI3fSIp6M8faxyF/mRqpfT22inG1Y3GLF9kNzDFw5H7GeN2N8t6JER++mreDpKfnrE01Geblj89FU8XdVOR9UP"
    "RuweJP/whccRh1kjDtMjDtMjno5SmB7xdJSnG5YwPeLpqnY6qmEwYv8ghYcvAo44zhpxnB5xnB7xdJTi9IinozzdsMTpEU9XtdNR"
    "jYMRhwcpPnwRccTb//ueMeNt+dSQ99nolCFL+2x0zpBl6Fn6bGzUUNdCVvu97KcdH8jOH8oOed7NvHk3MO8G5j2dpX02Pu/pLEPP"
    "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"
    "0gwizSDSDCzNOE+aEaQZQZqQpQjShCxDzxJBmlDXQlbjUJpBpBlFmpGlGedJM4I0I0gTshRBmpBl6FkiSBPqWshqHEozijSjSDOy"
    "NOM8aUaQZgRpQpYiSBOyDD1LBGlCXQtZjUNpRpFmFGlGlmacJ80I0owgTchSBGlClqFniSBNqGshq3EozSjSjCLNyNKM86QZQZoR"
    "pAlZiiBNyDL0LBGkCXUtZDUOpRlFmlGkGVmacZ40I0gzgjQhSxGkCVmGniWCNKGuhazGoTSjSDOKNCNLM86TZgRpRpAmZCmCNCHL"
    "0LNEkCbUtZDVOJRmFGlGkWZkacZ50owgzQjShCxFkCZkGXqWPhsdOEgTstrvpR+4SDOKNCNLM86TZgRpRpAmZCmCNCHL0LNEkCbU"
    "tZDVOJRmFGlGkWZkacZ50owgzQjShCxFkCZkGXqWCNKEuhayGofSjCLNKNKMLM3meB41d/VTI+/D0ZlTmPpwdOoUZmpbvoZjg6fK"
    "lsL6dUP97AWdsvZwu9Px6R/dXnTdJi02i2dPrrr1x+60u7y8PThbfbqW7v7wu9WDdfdh+wCfx8/V4dFP66f6cdZj683j2oytq8d1"
    "rM9J8/j52PtP9OPnY/2Lfvxyt370bfvPntysl9ebNzeb5er69uBitV7+tbreLC5Pu+tNt+7Onx7KnD53683y7IfFi25xvrz+eLt7"
    "8XG9PH+1vO5+ePWu212Acsybxcful8X641KOctl9kOXj3cW7vrtI715sVje76/X31UYu4N1ft0fp1ts32KYJTXOstFPqeHu3uA+r"
    "1WY82h9Pjv7p5kBOSba92J7g08PLxfX57dnipjs8uJE/1++Wf3XyEyaXgpxed3d9flhu3q+++/bsXv+2PN9c7F5uO79Z7zZ1vvpy"
    "/f6iu34jA5JdXy7O/ji5Pv/tYrnpdnM4Xy92Z3r43WDTzXL7oKLvpvpt5Wx1s9yOcDexoy+r9R+7a+7Z/wFQSwMEFAAAAAgAZIV7"
    "XTma5mvCBAAAOkMAAA0AAAB4bC9zdHlsZXMueG1s7Vxbr5s4EP4rCPV1FxISTqiSSC3SkfqyOlL7UKnqAwEnQQVMjXOU9NevbSAQ"
    "4smS65pzIA9gD5/n88z4huVMM7qL0Nc1QlTbxlGSzfQ1pelHw8j8NYq97G+cooRJlpjEHmVJsjKylCAvyDgojoyhadpG7IWJPp8m"
    "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"
    "AwAA0FYAAAAA"
)

HOJA_COMPRAS = "Compras"
FILA_INICIO_COMPRAS = 5   # la fila 4 son las cabeceras
FILA_MODELO_ESTILO = 98   # última fila formateada de la plantilla (se copia su estilo si hay más facturas)

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    client = None
    st.error("No se encontró la API KEY en los secretos. Configúrala en Streamlit Cloud.")

PROMPT = """
Analiza este documento PDF completo (puede tener varias páginas) como UNA SOLA factura.

INSTRUCCIONES CRÍTICAS SOBRE IMPUESTOS:
1. Si la factura tiene UN SOLO tipo de IVA: Extrae base, cuota y tipo normalmente.
2. Si la factura tiene VARIOS tipos de IVA (ej: 21% y 10%, o 21% y Exento/0%):
   DEBES devolver 'base_imponible', 'cuota_iva', 'fecha' y 'tipo_iva' como null (nulo).
   Sin embargo, SÍ debes extraer el 'nombre_proveedor', el 'numero_factura' y el 'total_factura'.

OTROS CAMPOS:
- 'nif_emisor': NIF de quien EMITE la factura, sin guiones, barras o puntos, solo las letras y numeros
- 'nombre_proveedor': Nombre fiscal de la empresa emisora.
- Fechas: DD/MM/AAAA.

Devuelve JSON estrictamente válido:
{
    "nif_emisor": "string",
    "fecha": "string" or null,
    "numero_factura": "string",
    "nombre_proveedor": "string",
    "base_imponible": float or null,
    "cuota_iva": float or null,
    "tipo_iva": float or null,
    "total_factura": float
}
"""


def procesar_con_gemini(uploaded_file, modelo_id):
    """Envía el PDF completo a Gemini y devuelve los datos de la factura."""
    try:
        pdf_part = types.Part.from_bytes(
            data=uploaded_file.getvalue(), mime_type="application/pdf"
        )
        response = client.models.generate_content(
            model=modelo_id,
            contents=[pdf_part, PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        datos = json.loads(response.text)
        if isinstance(datos, list):
            datos = datos[0]
        return datos
    except Exception as e:
        st.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
        return None


# --- HELPERS (sin dependencia de Streamlit) ---

def limpiar_nif(valor):
    """Deja solo letras y números en mayúsculas y quita el prefijo 'ES' si sobra."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    s = re.sub(r"[^A-Z0-9]", "", str(valor).upper())
    if s in ("NONE", "NAN", "NULL"):
        return ""
    if s.startswith("ES") and len(s) == 11:
        s = s[2:]
    return s


def _filas_crudas(archivo):
    """Lee el CSV/Excel del maestro como lista de filas (listas de texto), sin interpretar nada."""
    nombre = archivo.name.lower()
    contenido = archivo.getvalue()
    if nombre.endswith(".csv"):
        try:
            texto = contenido.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = contenido.decode("cp1252")
        delim = ";" if texto.count(";") >= texto.count(",") else ","
        return [[c.strip() for c in fila] for fila in csv.reader(io.StringIO(texto), delimiter=delim)]
    df = pd.read_excel(io.BytesIO(contenido), header=None, dtype=str).fillna("")
    return [[str(c).strip() for c in fila] for fila in df.values.tolist()]


def leer_maestro(archivo):
    """
    Lee el 'INFORME PLAN CONTABLE' tal cual sale del programa:
    busca solo la fila de cabeceras (Cuenta / NIF / Contrapartida), ignora las filas de
    arriba y la línea de guiones, y se queda con las cuentas que tienen NIF.
    Devuelve (df con NIF, Cuenta, Contrapartida, lista de NIF repetidos).
    """
    filas = _filas_crudas(archivo)
    idx_cab = None
    for i, fila in enumerate(filas):
        celdas = [c.lower() for c in fila]
        if "cuenta" in celdas and "nif" in celdas:
            idx_cab = i
            break
    if idx_cab is None:
        raise ValueError("No encuentro la fila de cabeceras (Cuenta, NIF, Contrapartida) en el fichero.")

    cab = [c.lower() for c in filas[idx_cab]]
    i_cuenta = cab.index("cuenta")
    i_nif = cab.index("nif")
    i_contra = cab.index("contrapartida") if "contrapartida" in cab else None

    registros = []
    for fila in filas[idx_cab + 1:]:
        def celda(i):
            return fila[i].strip() if i is not None and i < len(fila) else ""
        cuenta = re.sub(r"\.0$", "", celda(i_cuenta))
        nif = limpiar_nif(celda(i_nif))
        if not cuenta.isdigit() or not nif:
            continue  # salta guiones, cuentas de grupo sin NIF, líneas vacías...
        contra = re.sub(r"\.0$", "", celda(i_contra))
        registros.append({"NIF": nif, "Cuenta": cuenta, "Contrapartida": contra})

    df = pd.DataFrame(registros, columns=["NIF", "Cuenta", "Contrapartida"])
    repetidos = sorted(df.loc[df["NIF"].duplicated(keep=False), "NIF"].unique().tolist())
    df = df.drop_duplicates("NIF", keep="first")
    return df, repetidos


def _num(v):
    """Convierte a número (int si es entero) o None."""
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def _txt(v):
    if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s else None


def generar_excel(df_final):
    """Rellena la pestaña Compras de la plantilla y devuelve los bytes del .xlsx."""
    wb = openpyxl.load_workbook(io.BytesIO(base64.b64decode(PLANTILLA_B64)))
    ws = wb[HOJA_COMPRAS]

    for i, fila in enumerate(df_final.to_dict("records")):
        r = FILA_INICIO_COMPRAS + i
        if r > FILA_MODELO_ESTILO:  # más facturas que filas formateadas: copiar estilo
            for c in range(1, 11):
                ws.cell(r, c)._style = copy(ws.cell(FILA_MODELO_ESTILO, c)._style)

        ws.cell(r, 1).value = _txt(fila["FECHA"])
        ws.cell(r, 2).value = _num(fila["COD.PROVEED"])
        ws.cell(r, 3).value = _txt(fila["Nº FRA."])
        ws.cell(r, 4).value = _txt(fila["COMENTARIO"])
        ws.cell(r, 5).value = _num(fila["TOTAL FRA."])
        ws.cell(r, 6).value = _num(fila["COD. GASTOS"])
        ws.cell(r, 7).value = _num(fila["TIPO IVA"])
        ws.cell(r, 8).value = None
        ws.cell(r, 9).value = _num(fila["BASE IMP."])    # None => se borra la fórmula (varios IVA)
        ws.cell(r, 10).value = _num(fila["CUOTA IVA"])
        for c in (2, 6):  # códigos como número, no como texto
            ws.cell(r, c).number_format = "General"

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📄 Asesoría AI - Procesador de Facturas")

with st.expander("ℹ️ ¿Qué tengo que subir? (guía rápida)", expanded=True):
    st.markdown(
        """
**Necesitas 3 cosas:**

**1. Último número interno registrado** (campo numérico)
El último número de registro interno que ya habéis usado. La primera factura de este lote
será ese número **+ 1** y las siguientes irán correlativas.

**2. Informe Plan Contable (CSV o Excel)**
Súbelo **tal cual sale del programa de contabilidad** (*INFORME PLAN CONTABLE*), **sin editarlo**.
La app ignora sola las primeras filas, la línea de guiones y las cuentas sin NIF. Solo usa estas columnas:

| Columna | Para qué se usa |
|---|---|
| `NIF` | Para identificar al proveedor de cada factura |
| `Cuenta` | Sale como **COD.PROVEED** |
| `Contrapartida` | Sale como **COD. GASTOS** |

Si un proveedor no está en el informe, sus códigos salen **vacíos** (te avisamos al terminar).
Si un NIF aparece en dos cuentas, se usa la primera.

**3. Facturas en PDF**
Sube todos los PDFs del lote. **Cada PDF = una factura** (aunque tenga varias páginas).
No metas dos facturas distintas en el mismo PDF.

**Qué obtienes:** el Excel de *entrada de apuntes* con sus 4 pestañas (Ventas, Compras, Estandar y Banco 1).
**Solo se rellena Compras**; las demás quedan vacías.

⚠️ **Facturas con varios tipos de IVA** (ej. 21 % y 10 %): FECHA, TIPO IVA, BASE IMP. y CUOTA IVA
salen **vacíos a propósito** y hay que rellenarlos a mano. Sí salen el nº de factura, el proveedor y el total.
        """
    )

# BARRA LATERAL
with st.sidebar:
    st.header("Configuración")
    modelo_label = st.selectbox("Modelo AI", list(MODELOS.keys()))
    modelo = MODELOS[modelo_label]
    st.info(
        "Flash-Lite es el más barato y suele bastar para facturas normales. "
        "Si falla con facturas complejas o escaneadas, prueba Gemini 3.8 Flash."
    )

# ZONA PRINCIPAL
col1, col2 = st.columns(2)
with col1:
    ultimo_numero = st.number_input(
        "1️⃣ Último número interno registrado",
        min_value=0,
        value=1000,
        step=1,
        help="La primera factura de este lote será este número + 1",
    )
    archivo_maestro = st.file_uploader(
        "2️⃣ Informe Plan Contable (CSV o Excel, tal cual sale del programa)",
        type=["csv", "xlsx", "xls"],
    )

with col2:
    uploaded_files = st.file_uploader(
        "3️⃣ Facturas (PDF, una factura por archivo)",
        type=["pdf"],
        accept_multiple_files=True,
    )

if st.button("🚀 Procesar Facturas", type="primary"):
    if not uploaded_files:
        st.warning("⚠️ No has subido ningún archivo.")
    elif client is None:
        st.error("Falta la API KEY, no se puede procesar.")
    else:
        progreso = st.progress(0)
        estado = st.empty()

        datos_totales = []
        contador_interno = ultimo_numero
        total_docs = len(uploaded_files)

        for i, archivo in enumerate(uploaded_files):
            contador_interno += 1
            estado.write(f"Procesando {i+1}/{total_docs}: {archivo.name}...")

            datos = procesar_con_gemini(archivo, modelo)

            if datos:
                datos["numero_interno"] = contador_interno
                datos["archivo_origen"] = archivo.name
                datos_totales.append(datos)

            progreso.progress((i + 1) / total_docs)

        estado.write("¡Proceso completado! Generando Excel...")

        if datos_totales:
            df = pd.DataFrame(datos_totales)

            df["NIF"] = df["nif_emisor"].map(limpiar_nif) if "nif_emisor" in df.columns else ""

            for col in ["nombre_proveedor", "fecha", "numero_factura",
                        "base_imponible", "tipo_iva", "cuota_iva", "total_factura"]:
                if col not in df.columns:
                    df[col] = None

            df["COD.PROVEED"] = pd.NA
            df["COD. GASTOS"] = pd.NA

            if archivo_maestro is not None:
                try:
                    df_maestro, repetidos = leer_maestro(archivo_maestro)
                    df = df.drop(columns=["COD.PROVEED", "COD. GASTOS"])
                    df = pd.merge(df, df_maestro, on="NIF", how="left")
                    df = df.rename(columns={"Cuenta": "COD.PROVEED", "Contrapartida": "COD. GASTOS"})
                    if repetidos:
                        st.info("NIF con más de una cuenta en el informe (se usó la primera): " + ", ".join(repetidos))
                except Exception as e:
                    st.error(f"Error al leer el Informe Plan Contable: {e}")

            for col in ["COD.PROVEED", "COD. GASTOS"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            if archivo_maestro is not None:
                sin_cod = df[df["COD.PROVEED"].isna()]
                if not sin_cod.empty:
                    st.warning(
                        f"{len(sin_cod)} factura(s) sin código de proveedor (NIF no encontrado en el informe): "
                        + ", ".join(sin_cod["archivo_origen"].astype(str))
                    )

            df["SECCION"] = ""
            df["COMENTARIO"] = df.apply(
                lambda x: f"{x.get('numero_interno', '')} | {x.get('numero_factura', '')} | {x.get('nombre_proveedor', '')}",
                axis=1,
            )

            df = df.rename(columns={
                "fecha": "FECHA",
                "numero_factura": "Nº FRA.",
                "total_factura": "TOTAL FRA.",
                "tipo_iva": "TIPO IVA",
                "base_imponible": "BASE IMP.",
                "cuota_iva": "CUOTA IVA",
            })

            orden_final = [
                "FECHA", "COD.PROVEED", "Nº FRA.", "COMENTARIO", "TOTAL FRA.",
                "COD. GASTOS", "TIPO IVA", "SECCION", "BASE IMP.", "CUOTA IVA",
            ]
            df_final = df.reindex(columns=orden_final)

            st.success("Análisis y cruce completado.")
            st.dataframe(df_final)

            st.download_button(
                label="📥 Descargar Excel (entrada de apuntes)",
                data=generar_excel(df_final),
                file_name=f"entrada_de_apuntes_desde_{ultimo_numero + 1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("No se pudieron extraer datos de ningún archivo.")
