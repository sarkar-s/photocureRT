"""A python package to compute material development trajectory during photopolymerization
..moduleauthor:: Swarnavo Sarkar <swarnavo.sarkar@nist.gov>
"""

from InvertMaxwell import InvertMaxwell
from optparse import OptionParser

def process_options():
    usage = "usage: %prog [option1] arg1"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", dest="inputfile", help="File containing the resin parameters.", default="")
    
    [options, args] = parser.parse_args()
    
    if len(options.inputfile) != 0:
        InputFile = open(options.inputfile, 'r')
        try:
            InputFile = open(options.inputfile, 'r')
        except IOError:
            print >> sys.stderr , "ERROR : Cannot open inputfile. Check inputfile name."
            
        InputLines = InputFile.readlines()
        
        removeSet = []
        for l in InputLines:
            if l[0] == '#' or l[0] == '\n':
                removeSet.append(l)
        
        for rem in removeSet:
            InputLines.remove(rem)

        return InputLines

if __name__ == '__main__':
    input_lines = process_options()
    
    taumaker = InvertMaxwell(input_lines)
    taumaker.compute_eta()
    taumaker.compute_eta_exponent()
    taumaker.write_essential_results()
    taumaker.plot_results()
