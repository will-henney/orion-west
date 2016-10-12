for emline in 'nii', 'ha':
    rslt = []
    print('Cleaning', emline, 'ajustes data')
    with open('ajustes_{}.dat'.format(emline)) as f:
        stanzas = f.read().split('\n\n')
        print('Found {} stanzas'.format(len(stanzas)))
        col_labels = stanzas[0].split('\n')[0].split()
        rslt.append('\t'.join(col_labels))
        for stanza in stanzas[1:]:
            # skip first line since it is only metadata
            lines = stanza.split('\n') 
            for line in lines[1:]:
                values = line.split()
                # pad short lines
                values += '-'*(len(col_labels) - len(values))
                rslt.append('\t'.join(values))

    with open('ajustes_{}_cleaned.tab'.format(emline), 'w') as f:
        f.write('\n'.join(rslt))
