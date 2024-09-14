def investigate_species_over_time(
    start_data=datetime.datetime(2023, 5, 15, 0, 0, 0),
    no_days=500):
    
    date = start_data
    
    He_no_density = np.zeros(no_days)
    H_no_density = np.zeros(no_days)
    
    for i in range(no_days):
        date += datetime.timedelta(days=1)
        data = msise_flat(date, 550, 60, -70, 150, 150, 4)
        
        He_no_density[i] = data[0]
        H_no_density[i] = data[6]
    
    
    plt.plot(He_no_density)
    plt.plot(H_no_density)
    
    plt.yscale("log")
    
    plt.show()
    
    return 